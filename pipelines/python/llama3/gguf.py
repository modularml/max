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

from typing import Optional

from max.dtype import DType
from max.graph import Graph
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import GGUFWeights

from .model.hyperparameters import Hyperparameters
from nn import (
    Attention,
    Embedding,
    Linear,
    MLP,
    RMSNorm,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)
from .kv_cache_params import KVCacheParams


def feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: GGUFWeights,
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
    weights: GGUFWeights,
) -> Linear:
    return Linear(
        weights.weight.allocate(
            dtype, [in_features, out_features], quantization_encoding
        )
    )


def rms_norm(dims: int, eps: float, weights: GGUFWeights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.float32, [dims]), eps)


def embedding(
    params: Hyperparameters,
    vocab_size: int,
    hidden_dim: int,
    weights: GGUFWeights,
):
    return Embedding(
        weights.weight.allocate(
            params.dtype,
            [vocab_size, hidden_dim],
            params.quantization_encoding,
        )
    )


def attention(
    params: Hyperparameters, rope: RotaryEmbedding, weights: GGUFWeights
):
    return Attention(
        n_heads=params.n_heads,
        n_kv_heads=params.n_kv_heads,
        head_dim=params.head_dim,
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
    params: Hyperparameters,
    weights: GGUFWeights,
    _kv_params: KVCacheParams,
):
    # _kernel_names is currently unused
    # This is wired up with the expectation that we will
    # use it in a follow up PR which leverages the mo.opaque kv cache

    with graph:
        try:
            rope_scaling = weights.rope_freqs.weight.raw_tensor().data
        except KeyError:
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
            TransformerBlock(
                attention=attention(params, rope, weights.blk[i]),
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

        return Transformer(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            layers=layers,
            norm=rms_norm(
                params.hidden_dim,
                params.layer_norm_rms_epsilon,
                weights.output_norm,
            ),
            output=linear(
                params.dtype,
                params.quantization_encoding,
                params.vocab_size,
                params.hidden_dim,
                weights.output,
            ),
            theta=params.rope_theta,
            embedding=embedding(
                params, params.vocab_size, params.hidden_dim, weights.token_embd
            ),
        )
