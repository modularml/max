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
"""Contiguous KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

from typing import NewType, Union, List

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, MojoValue
from max.graph import Graph, TensorType, TensorValue, _OpaqueType, ops

from .cache_params import KVCacheParams
from .manager import KVCacheManager


class ContiguousKVCacheType(_OpaqueType):
    """Contiguous Mojo KV cache type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a contiguous KV cache."""
        super().__init__("ContiguousKVCache")


class ContiguousKVCacheCollectionType(_OpaqueType):
    """Collection of contiguous Mojo KV caches type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a KV cache collection."""
        super().__init__("ContiguousKVCacheCollection")


ContiguousKVCache = NewType("ContiguousKVCache", ContiguousKVCacheType)
ContiguousKVCacheCollection = NewType(
    "ContiguousKVCacheCollection", ContiguousKVCacheCollectionType
)


class FetchContiguousKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        key_cache: TensorValue,
        value_cache: TensorValue,
        cache_lengths: TensorValue,
        is_cache_empty: TensorValue,
    ) -> ContiguousKVCacheCollection:  # type: ignore
        """Constructs a ContiguousKVCacheCollection for use downstream."""

        key_shape = ops.shape_to_tensor(key_cache.shape)
        num_layers = key_shape[0]
        batch_size = key_shape[1]

        op_name = f"contiguous_kv_cache_collection_h{self.kv_params.n_kv_heads}_d{self.kv_params.head_dim}_bshd"
        return ops.custom(
            op_name,
            values=[
                key_cache,  # L, B, S, H, D
                value_cache,  # L, B, S, H, D
                cache_lengths,
                is_cache_empty,
                num_layers,
                batch_size,
            ],
            out_types=[ContiguousKVCacheCollectionType()],
        )[0]


class ContiguousKVCacheManager(KVCacheManager):
    def compile_fetch_graph(self, session: InferenceSession) -> Graph:
        # Create one-op fetch graph
        cache_lengths_type = TensorType(DType.uint32, ("batch_size",))
        seq_ids_type = TensorType(DType.int32, ("batch_size",))
        is_cache_empty_type = TensorType(DType.bool, (1,))

        cache_type = TensorType(
            self.params.dtype,
            self.symbolic_cache_shape,
        )
        int_scalar_type = TensorType(DType.int32, (1,))

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

        return session.load(fetch_graph)

    def fetch(
        self, seq_ids: list[int]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        active_batch_size = len(seq_ids)

        # Lookup table and seq_ids are redundant identical tensors.
        cache_lengths = Tensor.zeros((active_batch_size,), DType.uint32)
        is_cache_empty = True
        for i, seq_id in enumerate(seq_ids):
            if seq_id not in self.cache_lengths:
                raise ValueError(f"seq_id: {seq_id} not currently in cache.")

            cache_len = self.cache_lengths[seq_id]
            cache_lengths[i] = cache_len
            if cache_len != 0:
                is_cache_empty = False

        cache_lengths = cache_lengths.to(self.device)

        # Grab the first n elements we need from pre-allocated memory
        key_cache = self.blocks[0, 0 : len(seq_ids), :, :, :, :]
        value_cache = self.blocks[1, 0 : len(seq_ids), :, :, :, :]
        is_cache_empty_buf = (
            self.true_tensor if is_cache_empty else self.false_tensor
        )
        return (key_cache, value_cache, cache_lengths, is_cache_empty_buf)

    def block_shape(self, n_sequences: int) -> list[Union[str, int]]:
        return [
            2,
            n_sequences,
            self.num_layers,
            self.max_seq_len,
            self.params.n_kv_heads,
            self.params.head_dim,
        ]

    @property
    def symbolic_cache_shape(self) -> list[str]:
        """Helper function to provide symoblic cache shape for continguous caches.
        """
        return [
            "num_layers",
            "batch_size",
            "seq_len",
            "n_kv_heads",
            "head_dim",
        ]

    def input_symbols(self) -> List[TensorType]:
        return [
            # key_cache
            TensorType(
                self.params.dtype,
                shape=[
                    "num_layers",
                    "batch_size",
                    "max_seq_len",
                    "num_kv_heads",
                    "head_dim",
                ],
            ),
            # value_cache
            TensorType(
                self.params.dtype,
                shape=[
                    "num_layers",
                    "batch_size",
                    "max_seq_len",
                    "num_kv_heads",
                    "head_dim",
                ],
            ),
            # cache_lengths
            TensorType(DType.uint32, shape=["batch_size"]),
            # is_cache_empty
            TensorType(DType.bool, shape=[1]),
        ]
