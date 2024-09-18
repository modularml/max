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

from dataclasses import dataclass
from typing import List, Tuple, Union

from max.graph import OpaqueValue, TensorValue, ValueLike, ops

from .attention import Attention
from .embedding import Embedding
from .kernels import (
    ContiguousKVCacheType,
    KVCacheParams,
    key_cache_for_layer,
    value_cache_for_layer,
)
from .mlp import MLP, Linear
from .norm import RMSNorm
from .optimized_attention import OptimizedAttention


@dataclass
class TransformerBlock:
    """
    Stacks Attention, FeedForward, and RMSNorm layers
    into single transformer block.
    """

    attention: Union[Attention, OptimizedAttention]
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: ValueLike,
        attention_mask: ValueLike,
        k_cache: Union[ContiguousKVCacheType, ValueLike],
        v_cache: Union[ContiguousKVCacheType, ValueLike],
    ) -> Tuple[TensorValue, TensorValue, TensorValue]:
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(x), attention_mask, k_cache, v_cache
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h, k_cache_update, v_cache_update


@dataclass
class Transformer:
    """
    Transformer model consisting of TransformerBlock layers.
    """

    dim: int
    n_heads: int
    layers: List[TransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding

    def __call__(self, tokens, attention_mask, k_cache, v_cache):
        h = self.embedding(tokens)
        # Use the embeddings as ground truth for the activation dtype.
        activations_dtype = h.dtype
        kv_cache_dtype = k_cache.dtype

        k_cache_updates = []
        v_cache_updates = []
        for i in range(len(self.layers)):
            h, k_cache_layer_update, v_cache_layer_update = self.layers[i](
                h,
                attention_mask,
                ops.cast(k_cache[:, i], activations_dtype),
                ops.cast(v_cache[:, i], activations_dtype),
            )
            k_cache_updates.append(ops.transpose(k_cache_layer_update, 0, 1))
            v_cache_updates.append(ops.transpose(v_cache_layer_update, 0, 1))

        return (
            # Cast outputs back to the KV cache dtype, which may differ from
            # the activations dtype.
            ops.cast(self.output(self.norm(h)), kv_cache_dtype),
            ops.cast(ops.stack(k_cache_updates, axis=1), kv_cache_dtype),
            ops.cast(ops.stack(v_cache_updates, axis=1), kv_cache_dtype),
        )


@dataclass
class OptimizedTransformer:
    """
    Transformer model consisting of TransformerBlock layers.
    """

    dim: int
    n_heads: int
    layers: List[TransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding
    kv_params: KVCacheParams

    def __call__(
        self, tokens, attention_mask, kv_cache_collection: OpaqueValue
    ) -> TensorValue:
        h = self.embedding(tokens)

        for i in range(len(self.layers)):
            h, k_cache_layer_update, v_cache_layer_update = self.layers[i](
                h,
                attention_mask,
                key_cache_for_layer(self.kv_params, i, kv_cache_collection),
                value_cache_for_layer(self.kv_params, i, kv_cache_collection),
            )

        return self.output(self.norm(h))
