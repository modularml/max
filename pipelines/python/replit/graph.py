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
from max.pipelines import PipelineConfig
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
from max.pipelines.kv_cache import (
    KVCacheParams,
    KVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
)


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
        weight=weights.weight.allocate(DType.float32, [dims]),
        eps=eps,
    )


def _attention(
    pipeline_config: PipelineConfig,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
    layer_index: int,
) -> AttentionImpl:
    k_in_dim = kv_params.n_kv_heads * kv_params.head_dim
    v_in_dim = kv_params.n_kv_heads * kv_params.head_dim
    q_in_dim = pipeline_config.huggingface_config.d_model
    wqkv = TensorValue(
        weights.attn_qkv.weight.allocate(
            pipeline_config.dtype,
            [
                k_in_dim + v_in_dim + q_in_dim,
                pipeline_config.huggingface_config.d_model,
            ],
            pipeline_config.quantization_encoding.quantization_encoding,  # type: ignore
        )
    )

    return Attention(
        n_heads=pipeline_config.huggingface_config.n_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=Linear(
            weights.attn_output.weight.allocate(
                pipeline_config.dtype,
                [
                    pipeline_config.huggingface_config.d_model,
                    pipeline_config.huggingface_config.d_model,
                ],
                pipeline_config.quantization_encoding.quantization_encoding,  # type: ignore
            )
        ),
        layer_idx=ops.constant(layer_index, dtype=DType.uint32),
    )


def _transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
):
    with graph:
        # Initialize Attention.
        layers = [
            TransformerBlock(
                attention=_attention(
                    pipeline_config, weights.blk[i], kv_params, i
                ),
                mlp=_feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.quantization_encoding.quantization_encoding,  # type: ignore
                    pipeline_config.huggingface_config.d_model,
                    12288,
                    weights.blk[i],
                ),
                attention_norm=_lp_layer_norm(
                    pipeline_config.huggingface_config.d_model,
                    1e-5,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=_lp_layer_norm(
                    pipeline_config.huggingface_config.d_model,
                    1e-5,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.n_layers)
        ]

        # Initialize Shared Embedding Weights.
        shared_embedding_weight = weights.token_embd.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.d_model,
            ],
            pipeline_config.quantization_encoding.quantization_encoding,  # type: ignore
        )

        return Transformer(
            dim=pipeline_config.huggingface_config.d_model,
            n_heads=pipeline_config.huggingface_config.n_heads,
            layers=layers,
            norm=_lp_layer_norm(
                pipeline_config.huggingface_config.d_model,
                1e-5,
                weights.output_norm,
            ),
            output=Linear(shared_embedding_weight),
            embedding=Embedding(shared_embedding_weight),
            kv_params=kv_params,
            kv_collection_constructor=FetchContinuousBatchingKVCacheCollection(
                kv_params
            ),
            all_logits=pipeline_config.enable_echo,
        )


def _build_graph(
    pipeline_config: PipelineConfig,
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
    kv_cache_types = kv_manager.input_symbols()[0]

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
        model = _transformer(graph, pipeline_config, weights, kv_params)
        tokens, attention_mask, valid_lengths, *kv_cache_inputs = graph.inputs
        outputs = model(
            tokens=tokens,
            valid_lengths=valid_lengths,
            kv_cache_inputs=kv_cache_inputs,
            attention_mask=attention_mask.cast(pipeline_config.dtype),  # type: ignore
        )
        graph.output(*outputs)
        return graph
