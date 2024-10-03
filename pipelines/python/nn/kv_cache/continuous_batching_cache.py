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

from typing import List, NewType
import asyncio
from max.driver import Device
from max.engine import InferenceSession, MojoValue
from max.graph import (
    OpaqueType,
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


ContinuousBatchingKVCacheCollection = NewType(
    "ContinuousBatchingKVCacheCollection",
    ContinuousBatchingKVCacheCollectionType,
)


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
        max_cache_size: int,
        max_seq_len: int,
        num_layers: int,
        session: InferenceSession,
        device: Device,
    ) -> None:
        self.params = params
        self.max_cache_size = max_cache_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.device = device

        self.available = set(range(self.max_cache_size))
        self.semaphore = asyncio.BoundedSemaphore(self.max_cache_size)
        self.cache_lengths = {}  # type: ignore

    async def claim(self, n: int) -> List[int]:
        """Claims `n` blocks of memory in the cache for incoming requests.

        This returns a list of sequence ids, which identify a sequence's
        location within the cache. This sequence id can then be passed
        in the fetch function to return the ContinuousBatchingKVCacheCollection
        for those sequences.
        """
        seq_ids = []
        for _ in range(n):
            await self.semaphore.acquire()
            id = self.available.pop()
            seq_ids.append(id)
            self.cache_lengths[id] = 0

        return seq_ids

    def fetch(self, seq_ids: List[int]) -> MojoValue:
        raise NotImplementedError()

    def step(self, valid_lengths: dict[int, int]) -> None:
        """Update the `cache_lengths` objects to not that a new
        kv projection step has occured, and that the underlying memory
        has been written to. This `cache_lengths` value is then used
        downstream in `fetch` to track what section of memory should
        be used in the kernels.
        """

        for id, length in valid_lengths.items():
            if id not in self.cache_lengths:
                raise ValueError("seq_id: {id} not in cache.")

            self.cache_lengths[id] += length

    async def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """

        if seq_id not in self.cache_lengths:
            raise ValueError("`seq_id` provided not in cache.")

        self.available.add(seq_id)
        self.semaphore.release()

        del self.cache_lengths[seq_id]

    async def reset_cache(self) -> None:
        """A helper function to reset the entire cache."""
        for seq_id in self.cache_lengths:
            self.available.add(seq_id)
            self.semaphore.release()

        self.cache_lengths.clear()

    @property
    def slots_remaining(self) -> int:
        """The outstanding cache slots outstanding."""
        return self.semaphore._value
