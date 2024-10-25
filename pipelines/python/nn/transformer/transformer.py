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
from typing import Union

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from nn.kv_cache.continuous_batching_cache import (
    FetchContinuousBatchingKVCacheCollection,
)

from ..attention.interfaces import AttentionImpl
from ..embedding import Embedding
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    KVCacheParams,
)
from ..layer import Layer
from ..linear import MLP, Linear, FeedForward
from ..norm import RMSNorm, LPLayerNorm


@dataclass
class TransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: AttentionImpl
    mlp: Union[MLP, FeedForward]
    attention_norm: Union[RMSNorm, LPLayerNorm]
    mlp_norm: Union[RMSNorm, LPLayerNorm]

    def __call__(
        self,
        x: TensorValueLike,
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        valid_lengths: TensorValueLike,
        **kwargs,
    ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]:
        attn_out, kv_collection = self.attention(
            self.attention_norm(x),
            kv_collection,
            valid_lengths,
            **kwargs,
        )

        h = x + attn_out
        h = h + self.mlp(self.mlp_norm(h))

        return h, kv_collection


@dataclass
class Transformer(Layer):
    """Transformer model consisting for TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
    norm: Union[RMSNorm, LPLayerNorm]
    output: Linear
    embedding: Embedding
    kv_params: KVCacheParams
    kv_collection_constructor: FetchContinuousBatchingKVCacheCollection

    def __call__(
        self,
        tokens: TensorValueLike,
        valid_lengths: TensorValueLike,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> TensorValue:
        h = self.embedding(tokens)

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h, _ = layer(
                h,
                kv_collection,
                valid_lengths,
                **kwargs,
            )

        # Predict using the last non-pad token (right-padded).
        # `gather_nd` expects a static last dimension, so we unsqueeze.
        last_token = ops.gather_nd(
            h, indices=ops.unsqueeze(valid_lengths - 1, -1), batch_dims=1
        )
        # Always return float32 logits, no matter the activation type
        return ops.cast(self.output(self.norm(last_token)), DType.float32)
