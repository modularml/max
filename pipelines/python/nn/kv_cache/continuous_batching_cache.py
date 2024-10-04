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

import asyncio
from typing import List, NewType, Union

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, MojoValue
from max.graph import Graph, OpaqueType, TensorType, TensorValue, ops

from .cache_params import KVCacheLayout, KVCacheParams


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


class FetchContinuousBatchingKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 1],
        is_cache_empty: TensorValue,
        seq_ids: TensorValue,  # List[Int]
    ) -> MojoValue:
        """Constructs a ContinuousBatchingKVCacheCollection for use downstream.
        """

        # Explicit validation.
        if blocks.dtype != self.kv_params.dtype:
            msg = (
                f"expected blocks to be dtype: {self.kv_params.dtype}, got"
                f" {blocks.dtype}"
            )
            raise ValueError(msg)

        if blocks.rank != 6:
            msg = f"expected blocks to be of rank 6, got {blocks.rank}"
            raise ValueError(msg)

        # For all tensors other than the blocks tensor, the length should be equivalent
        # to batch size, which is unknown within the graph at this stage.
        if cache_lengths.dtype != DType.uint32:
            msg = (
                "expected cache lengths to be dtype: uint32, got"
                f" {cache_lengths.dtype}"
            )
            raise ValueError(msg)

        if lookup_table.dtype != DType.int32:
            msg = (
                "expected lookup_table to be dtype: int32, got"
                f" {lookup_table.dtype}"
            )
            raise ValueError(msg)

        if seq_ids.dtype != DType.int32:
            msg = f"expected seq_ids to be dtype: int32, got {seq_ids.dtype}"
            raise ValueError(msg)

        if cache_lengths.shape[0] != seq_ids.shape[0]:
            msg = (
                f"cache_lengths ({cache_lengths.shape[0]}) and"
                f" seq_ids ({seq_ids.shape[0]}) not the same shape."
            )
            raise ValueError(msg)

        if lookup_table.shape[0] != seq_ids.shape[0]:
            msg = (
                f"lookup_table ({lookup_table.shape[0]}) and"
                f" seq_ids ({seq_ids.shape[0]}) not the same shape."
            )
            raise ValueError(msg)

        op_name = f"continuous_batching_kv_cache_collection_h{self.kv_params.n_kv_heads}_d{self.kv_params.head_dim}_{self.kv_params.layout}"
        return ops.custom(
            op_name,
            values=[
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
                seq_ids,
            ],
            out_types=[ContinuousBatchingKVCacheCollectionType()],
        )[0]


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

        # Attributes for managing available slots.
        self.available = set(range(self.max_cache_size))
        self.semaphore = asyncio.BoundedSemaphore(self.max_cache_size)
        self.cache_lengths = {}  # type: ignore

        # Create one-op fetch graph
        blocks_type = TensorType(self.params.dtype, self.symbolic_block_shape)
        cache_lengths_type = TensorType(DType.uint32, ("batch_size",))
        lookup_table_type = TensorType(DType.int32, ("batch_size",))
        is_cache_empty_type = TensorType(DType.bool, (1,))
        seq_ids_type = TensorType(DType.int32, ("batch_size",))
        fetch_graph = Graph(
            "fetch_kv_collection",
            FetchContinuousBatchingKVCacheCollection(self.params),
            input_types=[
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                is_cache_empty_type,
                seq_ids_type,
            ],
        )

        self.fetch_model = session.load(fetch_graph)

        # Allocate cache memory
        block_shape = self.block_shape(self.max_cache_size)
        self.blocks = Tensor.zeros(
            block_shape, dtype=self.params.dtype, device=self.device
        )

        # Allocate true/false tensors.
        self.true_tensor = Tensor.zeros((1,), DType.bool)
        self.true_tensor[0] = True

        self.false_tensor = Tensor.zeros((1,), DType.bool)
        self.false_tensor[0] = False

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
        batch_size = len(seq_ids)

        # Grab seq_ids, cache_lengths, and lookup_table
        seq_ids_tensor = Tensor.zeros((batch_size,), DType.int32)
        cache_lengths = Tensor.zeros((batch_size,), DType.uint32)
        lookup_table = Tensor.zeros((batch_size,), DType.int32)
        is_cache_empty = True
        for i, seq_id in enumerate(seq_ids):
            if seq_id not in self.cache_lengths:
                raise ValueError(f"seq_id: {seq_id} not currently in cache.")

            seq_ids_tensor[i] = seq_id
            lookup_table[i] = seq_id
            cache_len = self.cache_lengths[seq_id]
            cache_lengths[i] = cache_len

            if cache_len != 0:
                is_cache_empty = False

        # Construct the KV Cache Collection by executing the fetch model
        # `blocks` should be on the execution device.
        return self.fetch_model.execute(
            self.blocks,
            cache_lengths,
            lookup_table,
            self.true_tensor if is_cache_empty else self.false_tensor,
            seq_ids_tensor,
        )[0]

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

    def block_shape(self, n_sequences: int) -> list[int]:
        """Get the shape of the cache for a given number of sequences."""
        if self.params.layout == KVCacheLayout.BHSD:
            return [
                n_sequences,
                2,
                self.num_layers,
                self.params.n_kv_heads,
                self.max_seq_len,
                self.params.head_dim,
            ]
        else:
            return [
                n_sequences,
                2,
                self.num_layers,
                self.max_seq_len,
                self.params.n_kv_heads,
                self.params.head_dim,
            ]

    @property
    def symbolic_block_shape(self) -> list[Union[str, int]]:
        """Get the symbolic shape of the blocks objects."""
        if self.params.layout == KVCacheLayout.BHSD:
            return [
                "batch_size",
                2,
                "num_layers",
                "n_kv_heads",
                "max_seq_len",
                "head_dim",
            ]
        else:
            return [
                "batch_size",
                2,
                "num_layers",
                "max_seq_len",
                "n_kv_heads",
                "head_dim",
            ]
