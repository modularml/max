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
from max.dtype import DType
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
        max_batch_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.keys = np.zeros(
            shape=(max_length, n_layers, max_batch_size, n_kv_heads, head_dim),
            dtype=np.float32,
        )
        self.values = np.zeros(
            shape=(max_length, n_layers, max_batch_size, n_kv_heads, head_dim),
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
        batch_size = new_keys.shape[2]

        self.keys[
            self.sequence_length : new_sequence_length, :, :batch_size, ...
        ] = new_keys
        self.values[
            self.sequence_length : new_sequence_length, :, :batch_size, ...
        ] = new_values
        self.sequence_length = new_sequence_length

    def keys_view(self, batch_size: int) -> npt.NDArray:
        """A view into the main key cache."""
        return self.keys[0 : self.sequence_length, :, :batch_size, ...]

    def values_view(self, batch_size: int) -> npt.NDArray:
        """A view into the main value cache."""
        return self.values[0 : self.sequence_length, :, :batch_size, ...]


class ContiguousKVCacheType(OpaqueType):
    """Contiguous Mojo KV cache type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a contiguous KV cache."""
        super().__init__("ContiguousKVCache")


class ContiguousKVCacheCollectionType(OpaqueType):
    """Collection of contiguous Mojo KV caches type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a KV cache collection."""
        super().__init__("ContiguousKVCacheCollection")


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
        is_cache_empty: TensorValue,
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
                key_cache,  # L, B, H, S, D: Layout Dependent (CPU vs GPU)
                value_cache,  # L, B, H, S, D: Layout Dependent (CPU vs GPU)
                cache_lengths,
                is_cache_empty,
                seq_ids,
                num_layers,
                batch_size,
            ],
            out_types=[ContiguousKVCacheCollectionType()],
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
        self.available = set(range(self.max_batch_size))
        # Mapping from sequence id to current KV cache length.
        self.cache_lengths: dict[int, int] = {}
        self.device = device

        # Create a graph for the fetch method.
        cache_type = TensorType(
            self.params.dtype,
            self.params.static_cache_shape,
        )
        cache_lengths_type = TensorType(DType.int32, ("batch_size",))
        seq_ids_type = TensorType(DType.int32, ("seq_len",))
        int_scalar_type = TensorType(DType.int32, (1,))
        is_cache_empty_type = TensorType(DType.bool, (1,))
        fetch_graph = Graph(
            "fetch_kv_collection",
            FetchContiguousKVCacheCollection(self.params),
            input_types=[
                cache_type,
                cache_type,
                cache_lengths_type,
                is_cache_empty_type,
                seq_ids_type,
                int_scalar_type,
                int_scalar_type,
            ],
        )

        self.fetch_model = session.load(fetch_graph)

        # Initialize Block Buffer.
        block_shape = self.block_shape(self.max_batch_size)
        self.blocks_buf = Tensor.zeros(
            block_shape, dtype=self.params.dtype, device=self.device
        )

    def claim(self, batch_size: int) -> List[int]:
        """Assign `batch_size` blocks for incoming requests.

        This returns a list of sequence_ids, which identifies a sequence's
        entries within the cache. These sequence_ids can then be used downstream
        to fetch specific KVCacheCollection objects.
        """

        if len(self.available) < batch_size:
            raise ValueError("no remaining slots available in kv cache")

        seq_ids = []
        for _ in range(batch_size):
            id = self.available.pop()
            seq_ids.append(id)
            self.cache_lengths[id] = 0

        return seq_ids

    def block_shape(self, n_sequences: int) -> list[int]:
        """Get the shape of the cache for a given number of sequences."""
        if self.params.layout == KVCacheLayout.BHSD:
            return [
                2,
                n_sequences,
                self.num_layers,
                self.params.n_kv_heads,
                self.max_seq_len,
                self.params.head_dim,
            ]
        else:
            return [
                2,
                n_sequences,
                self.num_layers,
                self.max_seq_len,
                self.params.n_kv_heads,
                self.params.head_dim,
            ]

    # [0, 1, 2] -> [1, 2, 3]
    def fetch(self, seq_ids: List[int]) -> ContiguousKVCacheCollection:
        """Retrieves the pre-assigned blocks for the given seq_ids."""

        # Grab the first n elements we need from `blocks_buf`.
        # B, L, H, S, D -> L, B, H, S, D: Layout dependent
        key_cache = self.blocks_buf[0, 0 : len(seq_ids), :, :, :, :]
        value_cache = self.blocks_buf[1, 0 : len(seq_ids), :, :, :, :]

        seq_ids_tensor = Tensor.zeros((len(seq_ids),), DType.int32)
        cache_lengths = Tensor.zeros((len(seq_ids),), DType.int32)
        is_cache_empty = True
        for i, seq_id in enumerate(seq_ids):
            seq_ids_tensor[i] = seq_id
            cache_len = self.cache_lengths[seq_id]
            cache_lengths[i] = cache_len
            if cache_len != 0:
                is_cache_empty = False

        is_cache_empty_tensor = Tensor.zeros((1,), DType.bool)
        is_cache_empty_tensor[0] = is_cache_empty

        # Call construct_kv_cache_collection.
        # Construct the KV cache collection by executing the fetch model.
        # `key_cache` and `value_cache` should be on the execution device.
        # All other arguments should be on the host.
        return self.fetch_model.execute(
            key_cache,
            value_cache,
            cache_lengths,
            is_cache_empty_tensor,
            seq_ids_tensor,
            np.array([self.num_layers]).astype(np.int32),
            np.array([len(seq_ids)]).astype(np.int32),
            copy_inputs_to_device=False,
        )[0]

    def step(self, valid_lengths: dict[int, int]) -> None:
        """Commits changes to the ContiguousKVCache blocks.

        This is used to note that a KV projection step has occured and
        the values in these buffers have been written to. We note the new tokens
        in the blocks and update the valid_length counter.
        """

        for k, v in valid_lengths.items():
            self.cache_lengths[k] += v

    def release(self, seq_id: int) -> None:
        """Marks `seq_id` as no longer necessary, their blocks are reintroduced
        to the pool.
        """
        self.available.add(seq_id)
        del self.cache_lengths[seq_id]

    def reset_cache(self) -> None:
        """Releases all existing seq_ids, to return the values to the pool."""
        for seq_id in self.cache_lengths:
            self.available.add(seq_id)

        self.cache_lengths.clear()
