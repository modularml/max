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
"""Build a Llama3 model via Graph API from GGUF weights."""

from typing import Optional, Union

from max.dtype import DType
from max.graph import Graph, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import Weights
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    OptimizedRotaryEmbedding,
    RMSNorm,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)
from nn.attention.attention_with_rope import AttentionWithRope

from .model.hyperparameters import Hyperparameters


def feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
):
    return MLP(
        linear(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.ffn_gate,
        ),
        linear(
            dtype,
            quantization_encoding,
            hidden_dim,
            feed_forward_length,
            weights.ffn_down,
        ),
        linear(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.ffn_up,
        ),
    )


def linear(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    in_features: int,
    out_features: int,
    weights: Weights,
) -> Linear:
    return Linear(
        weights.weight.allocate(
            dtype, [in_features, out_features], quantization_encoding
        )
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.float32, [dims]), eps)


def embedding(
    params: Hyperparameters,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
):
    return Embedding(
        weights.weight.allocate(
            params.dtype,
            [vocab_size, hidden_dim],
            params.quantization_encoding,
        )
    )


def _attention_opaque(kv_params, params, rope, weights, layer_idx):
    wq = ops.transpose(
        weights.attn_q.weight.allocate(
            params.dtype,
            [params.hidden_dim, params.hidden_dim],
            params.quantization_encoding,
        ),
        0,
        1,
    )
    wk = ops.transpose(
        weights.attn_k.weight.allocate(
            params.dtype,
            [params.kv_weight_dim, params.hidden_dim],
            params.quantization_encoding,
        ),
        0,
        1,
    )
    wv = ops.transpose(
        weights.attn_v.weight.allocate(
            params.dtype,
            [params.kv_weight_dim, params.hidden_dim],
            params.quantization_encoding,
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
            params.quantization_encoding,
            params.hidden_dim,
            params.hidden_dim,
            weights.attn_output,
        ),
        rope=rope,
        layer_idx=layer_idx,
    )


def _transformer_opaque(graph, params, weights, kv_params):
    with graph:
        try:
            rope_scaling = weights.rope_freqs.weight.raw_tensor().data
        except KeyError:
            # Set default RoPE scaling if the tensor isn't present in the GGUF
            # file.
            rope_scaling = None

        rope = OptimizedRotaryEmbedding(
            dim=params.hidden_dim,
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
                    weights.blk[i],
                    layer_idx=ops.constant(i, DType.uint32),
                ),
                mlp=feed_forward(
                    params.dtype,
                    params.quantization_encoding,
                    params.hidden_dim,
                    params.feed_forward_length,
                    weights.blk[i],
                ),
                attention_norm=rms_norm(
                    params.hidden_dim,
                    params.layer_norm_rms_epsilon,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    params.hidden_dim,
                    params.layer_norm_rms_epsilon,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(params.n_layers)
        ]

        embedding_layer = embedding(
            params,
            params.vocab_size,
            params.hidden_dim,
            weights.token_embd,
        )

        # Smaller model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if params.has_dedicated_output_weights:
            output = linear(
                params.dtype,
                params.quantization_encoding,
                params.vocab_size,
                params.hidden_dim,
                weights.output,
            )
        else:
            output = Linear(embedding_layer.weights)

        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        return Transformer(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            layers=layers,
            norm=rms_norm(
                params.hidden_dim,
                params.layer_norm_rms_epsilon,
                weights.output_norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
        )


def attention(
    kv_params: KVCacheParams,
    params: Hyperparameters,
    rope: Union[OptimizedRotaryEmbedding, RotaryEmbedding],
    weights: Weights,
):
    return NaiveAttentionWithRope(
        n_heads=params.n_heads,
        kv_params=kv_params,
        dim=params.hidden_dim,
        wk=linear(
            params.dtype,
            params.quantization_encoding,
            params.kv_weight_dim,
            params.hidden_dim,
            weights.attn_k,
        ),
        wv=linear(
            params.dtype,
            params.quantization_encoding,
            params.kv_weight_dim,
            params.hidden_dim,
            weights.attn_v,
        ),
        wq=linear(
            params.dtype,
            params.quantization_encoding,
            params.hidden_dim,
            params.hidden_dim,
            weights.attn_q,
        ),
        wo=linear(
            params.dtype,
            params.quantization_encoding,
            params.hidden_dim,
            params.hidden_dim,
            weights.attn_output,
        ),
        rope=rope,
    )


def transformer(
    graph: Graph,
    cache_strategy: KVCacheStrategy,
    params: Hyperparameters,
    weights: Weights,
    kv_params: KVCacheParams,
):
    if cache_strategy == KVCacheStrategy.CONTINUOUS:
        return _transformer_opaque(graph, params, weights, kv_params)

    with graph:
        try:
            rope_scaling = weights.rope_freqs.weight.raw_tensor().data
        except AttributeError:
            # Set default RoPE scaling if the tensor isn't present in the GGUF
            # file.
            rope_scaling = None

        rope = RotaryEmbedding(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            theta=params.rope_theta,
            max_seq_len=params.seq_len,
            rope_scaling=rope_scaling,
        )

        layers = [
            NaiveTransformerBlock(
                attention=attention(kv_params, params, rope, weights.blk[i]),
                mlp=feed_forward(
                    params.dtype,
                    params.quantization_encoding,
                    params.hidden_dim,
                    params.feed_forward_length,
                    weights.blk[i],
                ),
                attention_norm=rms_norm(
                    params.hidden_dim,
                    params.layer_norm_rms_epsilon,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    params.hidden_dim,
                    params.layer_norm_rms_epsilon,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(params.n_layers)
        ]

        embedding_layer = embedding(
            params, params.vocab_size, params.hidden_dim, weights.token_embd
        )

        # Smaller model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if params.has_dedicated_output_weights:
            output = linear(
                params.dtype,
                params.quantization_encoding,
                params.vocab_size,
                params.hidden_dim,
                weights.output,
            )
        else:
            output = Linear(embedding_layer.weights)

        return NaiveTransformer(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            layers=layers,
            norm=rms_norm(
                params.hidden_dim,
                params.layer_norm_rms_epsilon,
                weights.output_norm,
            ),
            output=output,
            theta=params.rope_theta,
            embedding=embedding_layer,
        )
