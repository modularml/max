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

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)

from ..attention.interfaces import AttentionImpl, AttentionImplQKV
from ..embedding import Embedding
from ..layer import Layer
from ..linear import MLP, Linear
from ..norm import LPLayerNorm, RMSNorm
from ..sequential import Sequential


@dataclass
class TransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: AttentionImpl | AttentionImplQKV
    mlp: MLP | Sequential
    attention_norm: RMSNorm | LPLayerNorm
    mlp_norm: RMSNorm | LPLayerNorm

    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        attn_out = self.attention(
            self.attention_norm(x), kv_collection, **kwargs
        )

        h = x + attn_out
        h = h + self.mlp(self.mlp_norm(h))

        return h


@dataclass
class Transformer(Layer):
    """Transformer model consisting for TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
    norm: RMSNorm | LPLayerNorm
    output: Linear
    embedding: Embedding
    kv_params: KVCacheParams
    kv_collection_constructor: FetchContinuousBatchingKVCacheCollection
    all_logits: bool = False

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embedding(tokens)

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        if self.all_logits:
            # When echo is enabled, the logits of the input tokens are
            # returned.
            logits = ops.cast(self.output(self.norm(h)), DType.float32)
            if "input_row_offsets" in kwargs:
                # For ragged tensors gather the last tokens from packed dim 0.
                input_row_offsets: TensorValueLike = kwargs["input_row_offsets"]
                last_token_indices = input_row_offsets[1:] - 1  # type: ignore
                last_token_logits = ops.gather(
                    logits, last_token_indices, axis=0
                )
            else:
                # For padded tensors, use `gather_nd`.
                # Unsqueeze since `gather_nd` expects a static last dim.
                valid_lengths: TensorValueLike = kwargs["valid_lengths"]
                last_token_logits = ops.gather_nd(
                    logits,
                    indices=ops.unsqueeze(valid_lengths - 1, -1),  # type: ignore
                    batch_dims=1,
                )
            return (last_token_logits, logits)
        else:
            # Otherwise, only return the logits for the last non-pad token
            # (right-padded).
            if "input_row_offsets" in kwargs:
                # For ragged tensors gather the last tokens from packed dim 0.
                input_row_offsets = kwargs["input_row_offsets"]
                last_token_indices = input_row_offsets[1:] - 1  # type: ignore
                # Should be: last_token = h[last_token_indices]
                last_token = ops.gather(h, last_token_indices, axis=0)
            else:
                # For padded tensors, use `gather_nd`.
                # Unsqueeze since `gather_nd` expects a static last dim.
                valid_lengths = kwargs["valid_lengths"]
                last_token = ops.gather_nd(
                    h,
                    indices=ops.unsqueeze(valid_lengths - 1, -1),  # type: ignore
                    batch_dims=1,
                )

            # Always return float32 logits, no matter the activation type
            return (
                ops.cast(self.output(self.norm(last_token)), DType.float32),
            )
