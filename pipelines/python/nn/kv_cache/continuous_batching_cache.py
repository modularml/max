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
"""Continuous Batching enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

from typing import List, TypeAlias
from max.driver import Device
from max.engine import InferenceSession
from max.graph import (
    OpaqueType,
    OpaqueValue,
)
from .cache_params import KVCacheParams


class ContinuousBatchingKVCacheCollectionType(OpaqueType):
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a continuous batching KV cache collection.
        """
        super().__init__("ContinuousBatchingKVCacheCollection")


ContinuousBatchingKVCacheCollection: TypeAlias = OpaqueValue


class ContinuousBatchingKVCacheManager:
    """Manages a Batch-split KV cache across multiple user sessions.

    Each request is assigned a seq_id, which is associated with a set of buffers
    to store the key and value projections per layer.

    The order of calls for an active request is expected to be:
    * claim -- assigned blocks to the sequence and give it a unique id
    * step -- commit context encoding projections
    * foreach token generated:
        * fetch -- retrieve blocks based on a seq_id
        * step -- commit token generation projections
    * release -- mark blocks as not in use

    TODO this is not currently threadsafe, make it so
    """

    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        session: InferenceSession,
        device: Device,
    ) -> None:
        self.params = params
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.device = device

    async def claim(self, n_slots: int) -> List[int]:
        raise NotImplementedError()

    def fetch(self, seq_ids: List[int]) -> ContinuousBatchingKVCacheCollection:
        raise NotImplementedError()

    def step(self, valid_lengths: dict[int, int]) -> None:
        raise NotImplementedError()

    async def release(self, seq_id: int) -> None:
        raise NotImplementedError()

    async def reset_cache(self) -> None:
        raise NotImplementedError()

    @property
    def slots_remaining(self) -> int:
        raise NotImplementedError()
