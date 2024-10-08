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

from .cache_params import KVCacheParams
from .manager import KVCacheManager


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

        if lookup_table.dtype != DType.uint32:
            msg = (
                "expected lookup_table to be dtype: uint32, got"
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

        op_name = f"continuous_batching_kv_cache_collection_h{self.kv_params.n_kv_heads}_d{self.kv_params.head_dim}_bshd"
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


class ContinuousBatchingKVCacheManager(KVCacheManager):
    def compile_fetch_graph(self, session: InferenceSession) -> Graph:
        # Create one-op fetch graph
        cache_lengths_type = TensorType(DType.uint32, ("batch_size",))
        lookup_table_type = TensorType(DType.uint32, ("batch_size",))
        seq_ids_type = TensorType(DType.int32, ("batch_size",))
        is_cache_empty_type = TensorType(DType.bool, (1,))

        blocks_type = TensorType(self.params.dtype, self.symbolic_block_shape)
        # seq_ids and lookup_table are directly redundant
        # one will be removed later on at the kernel level.
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

        return session.load(fetch_graph)

    def fetch(self, seq_ids: List[int]) -> MojoValue:
        active_batch_size = len(seq_ids)

        # Lookup table and seq_ids are redundant identical tensors.
        seq_ids_tensor = Tensor.zeros((active_batch_size,), DType.int32)
        lookup_table_tensor = Tensor.zeros((active_batch_size,), DType.uint32)
        cache_lengths = Tensor.zeros((active_batch_size,), DType.uint32)
        is_cache_empty = True
        for i, seq_id in enumerate(seq_ids):
            if seq_id not in self.cache_lengths:
                raise ValueError(f"seq_id: {seq_id} not currently in cache.")

            seq_ids_tensor[i] = seq_id
            lookup_table_tensor[i] = seq_id
            cache_len = self.cache_lengths[seq_id]
            cache_lengths[i] = cache_len
            if cache_len != 0:
                is_cache_empty = False

        # Cache Lengths buf has to be held on the object
        # and persisted beyond the fetch call, to ensure the object
        # is not destructed early, and the kernel can continue to
        # refer to this object.
        self.cache_lengths_buf = cache_lengths.to(self.device)
        self.lookup_table = lookup_table_tensor.to(self.device)

        # lookup_table and seq_ids are redundant identical tensors.
        return self.fetch_model.execute(
            self.blocks,
            self.cache_lengths_buf,
            self.lookup_table,
            self.true_tensor if is_cache_empty else self.false_tensor,
            seq_ids_tensor.to(self.device),
            copy_inputs_to_device=False,
        )[0]

    def block_shape(self, n_sequences: int) -> list[Union[str, int]]:
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
        """Get the symbolic shape of the cache memory allocation."""
        return [
            "batch_size",
            2,
            "num_layers",
            "max_seq_len",
            "n_kv_heads",
            "head_dim",
        ]
