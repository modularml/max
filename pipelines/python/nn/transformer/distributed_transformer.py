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
from typing import List

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)

from ..attention.interfaces import DistributedAttentionImpl
from ..embedding import Embedding
from ..layer import Layer
from ..linear import DistributedMLP, Linear
from ..norm import DistributedRMSNorm, LPLayerNorm, RMSNorm


# TODO (pavan): clean up duplicate instances of distribute_value, shard_col_value,
# shard_row_value across the codebase into a multi gpu utils file
def distribute_value(v, devices: List[DeviceRef]):
    return [v.to(device) for device in devices]


@dataclass
class DistributedTransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: DistributedAttentionImpl
    mlp: DistributedMLP
    attention_norm: DistributedRMSNorm
    mlp_norm: DistributedRMSNorm
    devices: List[DeviceRef]

    def __call__(
        self,
        xs: List[TensorValue],
        kv_collections: List[ContinuousBatchingKVCacheCollection],
        **kwargs,
    ) -> List[TensorValue]:
        attn_outs = self.attention(
            self.attention_norm(xs), kv_collections, **kwargs
        )

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs)]
        mlp_outs = self.mlp(self.mlp_norm(hs))
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs)]

        return hs


@dataclass
class DistributedTransformer(Layer):
    """Transformer model consisting for TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[DistributedTransformerBlock]
    norm: RMSNorm | LPLayerNorm
    output: Linear
    embedding: Embedding
    kv_params: KVCacheParams
    kv_collection_constructor: FetchContinuousBatchingKVCacheCollection
    devices: List[DeviceRef]
    all_logits: bool = False

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs_per_dev: List[
            tuple[TensorValue, TensorValue, TensorValue, TensorValue]
        ],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        embedding_lookup = self.embedding(tokens)
        h = distribute_value(embedding_lookup, self.devices)

        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collections, **kwargs)

        h0 = h[0]  # All the outputs are the same here.
        if self.all_logits:
            # When echo is enabled, the logits of the input tokens are
            # returned.
            logits = ops.cast(self.output(self.norm(h0)), DType.float32)
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
                last_token = ops.gather(h0, last_token_indices, axis=0)
            else:
                # For padded tensors, use `gather_nd`.
                # Unsqueeze since `gather_nd` expects a static last dim.
                valid_lengths = kwargs["valid_lengths"]
                last_token = ops.gather_nd(
                    h0,
                    indices=ops.unsqueeze(valid_lengths - 1, -1),  # type: ignore
                    batch_dims=1,
                )

            # Always return float32 logits, no matter the activation type
            return (
                ops.cast(self.output(self.norm(last_token)), DType.float32),
            )