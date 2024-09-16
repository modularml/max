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
"""KV cache for the Transformer."""

from __future__ import annotations

from typing import List, TypeAlias

import numpy as np
import numpy.typing as npt
from max.driver import Device, Tensor
from max.engine import InferenceSession
from max.graph import (
    Graph,
    OpaqueType,
    OpaqueValue,
    TensorType,
    TensorValue,
    ops,
)

from .kv_cache_params import KVCacheLayout, KVCacheParams


class KVCache:
    keys: npt.NDArray
    values: npt.NDArray
    sequence_length: int

    def __init__(
        self,
        max_length: int,
        batch_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.keys = np.zeros(
            shape=(max_length, n_layers, batch_size, n_kv_heads, head_dim),
            dtype=np.float32,
        )
        self.values = np.zeros(
            shape=(max_length, n_layers, batch_size, n_kv_heads, head_dim),
            dtype=np.float32,
        )
        self.sequence_length = 0

    def update(self, new_keys: npt.NDArray, new_values: npt.NDArray):
        """Insert the updated key and value cache elements in the main cache."""
        key_length = new_keys.shape[0]
        new_sequence_length = self.sequence_length + key_length
        assert new_sequence_length <= self.keys.shape[0], (
            f"kv-cache overflow, desired: {new_sequence_length}, "
            f"max: {self.keys.shape[0]}"
        )
        self.keys[self.sequence_length : new_sequence_length, ...] = new_keys
        self.values[
            self.sequence_length : new_sequence_length, ...
        ] = new_values
        self.sequence_length = new_sequence_length

    def keys_view(self) -> npt.NDArray:
        """A view into the main key cache."""
        return self.keys[0 : self.sequence_length, ...]

    def values_view(self) -> npt.NDArray:
        """A view into the main value cache."""
        return self.values[0 : self.sequence_length, ...]


ContiguousKVCacheType: TypeAlias = OpaqueType("ContiguousKVCache")
ContiguousKVCacheCollectionType: TypeAlias = OpaqueType(
    "ContiguousKVCacheCollection"
)
ContiguousKVCache: TypeAlias = OpaqueValue
ContiguousKVCacheCollection: TypeAlias = OpaqueValue


class FetchContiguousKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        key_cache: TensorValue,
        value_cache: TensorValue,
        cache_lengths: TensorValue,
        seq_ids: TensorValue,
        num_layers: TensorValue,
        batch_size: TensorValue,
    ) -> ContiguousKVCacheCollection:
        """Constructs an initial ContiguousKVCacheCollection for use downstream.
        """
        op_name = f"contiguous_kv_cache_collection_h{self.kv_params.n_kv_heads}_d{self.kv_params.head_dim}_{self.kv_params.layout}"
        return ops.custom(
            op_name,
            values=[
                key_cache,
                value_cache,
                cache_lengths,
                seq_ids,
                num_layers,
                batch_size,
            ],
            out_types=[ContiguousKVCacheCollectionType],
        )[0]


class ContiguousKVCacheManager:
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
        self.available = set([i for i in range(0, self.max_batch_size)])
        self.cache_lengths = {}
        self.device = device

        # Create a graph for the fetch method.
        cache_type = TensorType(
            self.params.dtype,
            ("start_pos", "n_layers", "batch_size", "n_kv_heads", "head_dim"),
        )
        cache_lengths_type = TensorType(self.params.dtype, (max_batch_size,))
        seq_ids_type = TensorType(self.params.dtype, ("seq_len",))
        int_scalar_type = TensorType(self.params.dtype, (1,))

        fetch_graph = Graph(
            "fetch_kv_collection",
            FetchContiguousKVCacheCollection(self.params),
            input_types=[
                cache_type,
                cache_type,
                cache_lengths_type,
                seq_ids_type,
                int_scalar_type,
                int_scalar_type,
            ],
        )

        self.fetch = session.load(fetch_graph)

        # Initialize Block Buffer.
        block_shape = [2] + self.cache_shape(self.max_batch_size)
        self.blocks_buf = Tensor.from_numpy(
            np.zeros(block_shape, dtype=np.float32)
        ).copy_to(self.device)

    def claim(self, batch_size: int) -> List[int]:
        """Assign `batch_size` blocks for incoming requests.

        This returns a list of sequence_ids, which identifies a sequence's
        entries within the cache. These sequence_ids can then be used downstream
        to fetch specific KVCacheCollection objects.
        """

        if len(self.available) < batch_size:
            raise ValueError("batch size too large")

        seq_ids = []
        for _ in range(batch_size):
            id = self.available.pop()
            seq_ids.append(id)
            self.cache_lengths[id] = 0

        return seq_ids

    def cache_shape(self, n_sequences: int) -> list[int]:
        """Get the shape of the cache for a given number of sequences."""
        if self.params.layout == KVCacheLayout.BHSD:
            return [
                self.num_layers,
                n_sequences,
                self.params.n_kv_heads,
                self.max_seq_len,
                self.params.head_dim,
            ]
        else:
            return [
                self.num_layers,
                n_sequences,
                self.max_seq_len,
                self.params.n_kv_heads,
                self.params.head_dim,
            ]

    # [0, 1, 2] -> [1, 2, 3]
    def fetch(self, seq_ids: List[int]) -> ContiguousKVCacheCollection:
        """Retrieves the pre-assigned blocks for the given seq_ids.

        and error is raised.
        if any of the seq_ids are not valid (e.g. no assigned blocks) then
        """

        cache_shape = self.cache_shape(len(seq_ids))
        # This is just grabbing the first n elements of memory we need
        key_cache = self.blocks_buf[0, 0 : len(seq_ids), :, :, :, :, :].reshape(
            cache_shape
        )
        value_cache = self.blocks_buf[1, 0 : len(seq_ids), :, :, :, :].reshape(
            cache_shape
        )
        cache_lengths = Tensor.from_numpy(
            [seq.l for seq in self.cache_lengths.values()]
        )
        seq_ids = Tensor.from_numpy(seq_ids)

        # Call construct_kv_cache_collection
        return self.fetch._execute(
            key_cache,
            value_cache,
            cache_lengths,
            seq_ids,
            self.num_layers,
            len(seq_ids),
        )

    def step(self, valid_lengths: dict[int, int]) -> None:
        """Commits changes to the ContiguousKVCache blocks.

        This is used to note that a KV projection step has occured and
        the values in these buffers have been written to. We note the new tokens
        in the blocks and update the valid_length counter.
        """

        if len(valid_lengths) != len(self.cache_lengths):
            raise ValueError(
                "Invalid valid_lengths passed, expected to match requests batch"
                " size."
            )

        for k, v in valid_lengths.items():
            self.cache_lengths[k] += v

    def release(self, seq_id: int) -> None:
        """Marks `seq_id` as no longer necessary, their blocks are reintroduced
        to the pool.
        """
        self.available.add(seq_id)
        del self.cache_lengths[seq_id]
