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
"""Build a Mistral model via Graph API from Safetensor weights."""

from max.dtype import DType
from max.graph import ops, Graph
from max.graph.weights import SafetensorWeights
from nn import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    OptimizedRotaryEmbedding,
    AttentionWithRope,
    TransformerBlock,
    Transformer,
)
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)

from .hyperparameters import Hyperparameters


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return MLP(
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.w1,
        ),
        linear(
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.feed_forward.w2,
        ),
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.w3,
        ),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


def rms_norm(dims: int, eps: float, weights: SafetensorWeights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.bfloat16, [dims]), eps)


def embedding(
    params: Hyperparameters,
    vocab_size: int,
    hidden_dim: int,
    weights: SafetensorWeights,
):
    return Embedding(
        weights.weight.allocate(
            params.dtype,
            [vocab_size, hidden_dim],
        )
    )


def _attention_opaque(
    kv_params: KVCacheParams,
    params: Hyperparameters,
    rope: OptimizedRotaryEmbedding,
    weights: SafetensorWeights,
    layer_idx: int,
):
    wq = ops.transpose(
        weights.attention.wq.weight.allocate(
            params.dtype,
            [params.n_heads * params.head_dim, params.hidden_dim],
        ),
        0,
        1,
    )
    wk = ops.transpose(
        weights.attention.wk.weight.allocate(
            params.dtype,
            [params.kv_weight_dim, params.hidden_dim],
        ),
        0,
        1,
    )
    wv = ops.transpose(
        weights.attention.wv.weight.allocate(
            params.dtype,
            [params.kv_weight_dim, params.hidden_dim],
        ),
        0,
        1,
    )
    wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

    return AttentionWithRope(
        n_heads=params.n_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            params.dtype,
            params.hidden_dim,
            params.n_heads * params.head_dim,
            weights.attention.wo,
        ),
        rope=rope,
        layer_idx=layer_idx,
    )


def transformer(
    graph: Graph,
    params: Hyperparameters,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
):
    with graph:
        try:
            rope_scaling = weights.rope_freqs.weight.raw_tensor().data
        except AttributeError:
            # Set default RoPE scaling if the tensor isn't present in the GGUF
            # file.
            rope_scaling = None

        rope = OptimizedRotaryEmbedding(
            dim=params.n_heads * params.head_dim,
            n_heads=params.n_heads,
            theta=params.rope_theta,
            max_seq_len=params.seq_len,
            rope_scaling=rope_scaling,
        )

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    params,
                    rope,
                    weights.layers[i],
                    layer_idx=ops.constant(i, DType.uint32),
                ),
                mlp=feed_forward(
                    params.dtype,
                    params.hidden_dim,
                    params.feed_forward_length,
                    weights.layers[i],
                ),
                attention_norm=rms_norm(
                    params.hidden_dim,
                    params.layer_norm_rms_epsilon,
                    weights.layers[i].attention_norm,
                ),
                mlp_norm=rms_norm(
                    params.hidden_dim,
                    params.layer_norm_rms_epsilon,
                    weights.layers[i].ffn_norm,
                ),
            )
            for i in range(params.n_layers)
        ]

        embedding_layer = embedding(
            params,
            params.vocab_size,
            params.hidden_dim,
            weights.tok_embeddings,
        )

        output = Linear(embedding_layer.weights)

        kv_collection_cls = FetchContinuousBatchingKVCacheCollection

        return Transformer(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            layers=layers,
            norm=rms_norm(
                params.hidden_dim,
                params.layer_norm_rms_epsilon,
                weights.norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
        )
