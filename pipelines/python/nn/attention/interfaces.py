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
"""General interface for Attention."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike

from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    KVCacheParams,
    KVCacheStrategy,
)
from ..layer import Layer
from ..linear import Linear


@dataclass
class AttentionImpl(ABC, Layer):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:
        - AttentionWithRope
        - AttentionWithAlibi
        - VanillaAttentionWithCausalMask
        - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    @dataclass
    class AttentionWithRope(AttentionImpl):
        rope: OptimizedRotaryEmbedding
        ...

    We expect the `__call__` abstractmethod to remain relatively consistent, however the **kwargs
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    @dataclass
    class VanillaAttentionWithCausalMask(AttentionImpl):
        ...

        def __call__(
            self,
            x: TensorValueLike,
            kv_collection: ContinuousBatchingKVCacheCollectionType,
            valid_lengths: TensorValueLike,
            **kwargs,
        ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

            if "attn_mask" not in kwargs:
                raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

            # Which we can then use the attention mask downstream like so:
            op(
                attn_mask = kwargs["attn_mask"]
            )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    layer_idx: TensorValue
    """The layer number associated with this Attention block."""

    wqkv: TensorValue
    """A concatenated weight vector with both q, k and v weights."""

    wo: Linear
    """A linear layer for the output projection."""

    def __post_init__(self) -> None:
        if self.kv_params.cache_strategy == KVCacheStrategy.NAIVE:
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        x: TensorValueLike,
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        valid_lengths: TensorValueLike,
        **kwargs,
    ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]:
        ...