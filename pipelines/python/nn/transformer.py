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
from typing import TYPE_CHECKING, Union

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops

from .kernels import key_cache_for_layer, value_cache_for_layer
from .kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
)
from .layer import Layer

if TYPE_CHECKING:
    from .attention import Attention
    from .embedding import Embedding
    from .kv_cache import (
        ContinuousBatchingKVCache,
        ContinuousBatchingKVCacheType,
        ContinuousBatchingKVCacheCollectionType,
        KVCacheParams,
    )
    from .mlp import MLP, Linear
    from .norm import RMSNorm
    from .optimized_attention import OptimizedAttention


@dataclass
class TransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: Attention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: TensorValueLike,
        attention_mask: TensorValueLike,
        k_cache: ContinuousBatchingKVCacheType | TensorValueLike,
        v_cache: ContinuousBatchingKVCacheType | TensorValueLike,
        start_pos: TensorValue,
        layer_index: int,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        attention_out = self.attention(
            self.attention_norm(x),
            attention_mask,
            k_cache,
            v_cache,
            start_pos,
            layer_index,
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h


@dataclass
class Transformer(Layer):
    """Transformer model consisting of TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding

    def __call__(self, tokens, attention_mask, k_cache, v_cache, start_pos):
        h = self.embedding(tokens)
        # Use the embeddings as ground truth for the activation dtype.
        activations_dtype = h.dtype
        kv_cache_dtype = k_cache.dtype

        for i in range(len(self.layers)):
            h = self.layers[i](
                h,
                attention_mask,
                k_cache,
                v_cache,
                start_pos,
                i,
            )

        seq_len = TensorValue(tokens.shape[1])
        return (
            # Cast outputs back to the KV cache dtype, which may differ from
            # the activations dtype.
            ops.cast(self.output(self.norm(h)), kv_cache_dtype),
            start_pos + seq_len,
        )


@dataclass
class OptimizedTransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: OptimizedAttention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: TensorValueLike,
        attention_mask: TensorValueLike,
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        k_cache: ContinuousBatchingKVCacheType | TensorValueLike,
        v_cache: ContinuousBatchingKVCacheType | TensorValueLike,
        valid_lengths: TensorValue,
    ) -> tuple[
        TensorValue, ContinuousBatchingKVCache, ContinuousBatchingKVCache
    ]:
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(x),
            attention_mask,
            kv_collection,
            k_cache,
            v_cache,
            valid_lengths,
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h, k_cache_update, v_cache_update


@dataclass
class OptimizedTransformer(Layer):
    """Transformer model consisting of OptimizedTransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[OptimizedTransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding
    kv_params: KVCacheParams
    kv_collection_constructor: FetchContinuousBatchingKVCacheCollection

    def __call__(
        self,
        tokens: TensorValue,
        attention_mask: TensorValue,
        valid_lengths: TensorValue,
        kv_cache_params: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
    ) -> TensorValue:
        h = self.embedding(tokens)

        kv_cache_collection = self.kv_collection_constructor(*kv_cache_params)

        for i, layer in enumerate(self.layers):
            h, _, _ = layer(
                h,
                attention_mask,
                kv_cache_collection,
                key_cache_for_layer(self.kv_params, i, kv_cache_collection),
                value_cache_for_layer(self.kv_params, i, kv_cache_collection),
                valid_lengths,
            )

        # Predict using the last non-pad token (right-padded).
        # `gather_nd` expects a static last dimension, so we unsqueeze.
        last_token = ops.gather_nd(
            h, indices=ops.unsqueeze(valid_lengths - 1, -1), batch_dims=1
        )
        # Always return float32 logits, no matter the activation type
        return ops.cast(self.output(self.norm(last_token)), DType.float32)
