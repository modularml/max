# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from typing import Optional

from max.dtype import DType
from max.graph import Graph, TensorType, TensorValue, ops
from max.graph.weights import GGUFWeights
from max.graph.quantization import QuantizationEncoding
from nn import (
    AttentionImpl,
    Attention,
    Embedding,
    Linear,
    LPLayerNorm,
    Sequential,
    Transformer,
    TransformerBlock,
)
from nn.kv_cache import (
    KVCacheParams,
    KVCacheManager,
)
from nn.kv_cache.continuous_batching_cache import (
    FetchContinuousBatchingKVCacheCollection,
)
from .hyperparameters import Hyperparameters


def _feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    input_dim: int,
    hidden_dim: int,
    weights: GGUFWeights,
):
    return Sequential(
        layers=[
            Linear(
                weights.ffn_up.weight.allocate(
                    dtype, [hidden_dim, input_dim], quantization_encoding
                )
            ),
            ops.gelu,
            Linear(
                weights.ffn_down.weight.allocate(
                    dtype, [input_dim, hidden_dim], quantization_encoding
                )
            ),
        ]
    )


def _lp_layer_norm(dims: int, eps: float, weights: GGUFWeights) -> LPLayerNorm:
    return LPLayerNorm(
        weight=weights.weight.allocate(DType.float32, [dims]), eps=eps
    )


def _attention(
    hyperparameters: Hyperparameters,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
    layer_index: int,
) -> AttentionImpl:
    k_in_dim = kv_params.n_kv_heads * kv_params.head_dim
    v_in_dim = kv_params.n_kv_heads * kv_params.head_dim
    q_in_dim = hyperparameters.hidden_dim
    wqkv = TensorValue(
        weights.attn_qkv.weight.allocate(
            hyperparameters.dtype,
            [k_in_dim + v_in_dim + q_in_dim, hyperparameters.hidden_dim],
            hyperparameters.quantization_encoding,
        )
    )

    return Attention(
        n_heads=hyperparameters.n_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=Linear(
            weights.attn_output.weight.allocate(
                hyperparameters.dtype,
                [hyperparameters.hidden_dim, hyperparameters.hidden_dim],
                hyperparameters.quantization_encoding,
            )
        ),
        layer_idx=ops.constant(layer_index, dtype=DType.uint32),
    )


def _transformer(
    graph: Graph,
    hyperparameters: Hyperparameters,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
):
    with graph:
        # Initialize Attention.
        layers = [
            TransformerBlock(
                attention=_attention(
                    hyperparameters, weights.blk[i], kv_params, i
                ),
                mlp=_feed_forward(
                    hyperparameters.dtype,
                    hyperparameters.quantization_encoding,
                    hyperparameters.hidden_dim,
                    12288,
                    weights.blk[i],
                ),
                attention_norm=_lp_layer_norm(
                    hyperparameters.hidden_dim,
                    hyperparameters.layer_norm_epsilon,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=_lp_layer_norm(
                    hyperparameters.hidden_dim,
                    hyperparameters.layer_norm_epsilon,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(hyperparameters.num_layers)
        ]

        # Initialize Shared Embedding Weights.
        shared_embedding_weight = weights.token_embd.weight.allocate(
            hyperparameters.dtype,
            [hyperparameters.vocab_size, hyperparameters.hidden_dim],
            hyperparameters.quantization_encoding,
        )

        return Transformer(
            dim=hyperparameters.hidden_dim,
            n_heads=hyperparameters.n_heads,
            layers=layers,
            norm=_lp_layer_norm(
                hyperparameters.hidden_dim,
                hyperparameters.layer_norm_epsilon,
                weights.output_norm,
            ),
            output=Linear(shared_embedding_weight),
            embedding=Embedding(shared_embedding_weight),
            kv_params=kv_params,
            kv_collection_constructor=FetchContinuousBatchingKVCacheCollection(
                kv_params
            ),
        )


def _build_graph(
    hyperparameters: Hyperparameters,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    # Graph input types.
    tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
    attn_mask_type = TensorType(
        DType.float32,
        shape=["batch_size", "n_heads", "seq_len", "post_seq_len"],
    )
    valid_lengths_type = TensorType(DType.uint32, shape=["batch_size"])
    kv_cache_types = kv_manager.input_symbols()

    # Initialize Graph.
    with Graph(
        "replit",
        input_types=[
            tokens_type,
            attn_mask_type,
            valid_lengths_type,
            *kv_cache_types,
        ],
    ) as graph:
        model = _transformer(graph, hyperparameters, weights, kv_params)
        tokens, attention_mask, valid_lengths, *kv_cache_inputs = graph.inputs
        logits = model(
            tokens=tokens,
            valid_lengths=valid_lengths,
            kv_cache_inputs=kv_cache_inputs,
            attention_mask=attention_mask,
        )
        graph.output(logits)
        return graph


def _argmax_sampler(dtype: DType):
    logits_type = TensorType(dtype, ["batch", "vocab_size"])
    return Graph("argmax", ops.argmax, input_types=[logits_type])
