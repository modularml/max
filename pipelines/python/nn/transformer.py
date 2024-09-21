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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from max.dtype import DType
from max.graph import OpaqueValue, TensorValue, ValueLike, ops

from .kernels import (
    ContiguousKVCacheCollection,
    ContiguousKVCacheType,
    KVCacheParams,
    key_cache_for_layer,
    kv_cache_length,
    value_cache_for_layer,
)

if TYPE_CHECKING:
    from .attention import Attention
    from .embedding import Embedding
    from .mlp import MLP, Linear
    from .norm import RMSNorm
    from .optimized_attention import OptimizedAttention


@dataclass
class TransformerBlock:
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: Attention | OptimizedAttention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: ValueLike,
        attention_mask: ValueLike,
        k_cache: ContiguousKVCacheType | ValueLike,
        v_cache: ContiguousKVCacheType | ValueLike,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(x), attention_mask, k_cache, v_cache
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h, k_cache_update, v_cache_update


@dataclass
class Transformer:
    """Transformer model consisting of TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
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
class OptimizedTransformerBlock:
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: Attention | OptimizedAttention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: ValueLike,
        attention_mask: ValueLike,
        k_cache: ContiguousKVCacheType | ValueLike,
        v_cache: ContiguousKVCacheType | ValueLike,
        start_pos: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(x), attention_mask, k_cache, v_cache, start_pos
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h, k_cache_update, v_cache_update


@dataclass
class OptimizedTransformer:
    """Transformer model consisting of OptimizedTransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[OptimizedTransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding
    kv_params: KVCacheParams

    def __call__(
        self,
        tokens,
        attention_mask,
        kv_cache_collection: ContiguousKVCacheCollection,
    ) -> TensorValue:
        h = self.embedding(tokens)

        # Plumb in the `start_pos` (previous sequence length), needed to
        # construct the attention mask.
        start_pos = kv_cache_length(self.kv_params, kv_cache_collection)
        for i, layer in enumerate(self.layers):
            h, _, _ = layer(
                h,
                attention_mask,
                key_cache_for_layer(self.kv_params, i, kv_cache_collection),
                value_cache_for_layer(self.kv_params, i, kv_cache_collection),
                start_pos,
            )

        return ops.cast(self.output(self.norm(h)), DType.float32)
