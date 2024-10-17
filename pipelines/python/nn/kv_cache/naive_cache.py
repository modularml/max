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
"""Naive KV cache for the Transformer."""

from typing import List
from max.driver import Device, Tensor
from max.dtype import DType
from max.graph import TensorType, BufferType
from .manager import KVCacheManager
from .cache_params import KVCacheParams


class NaiveKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        device: Device,
    ) -> None:
        super().__init__(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            device=device,
        )

        self.keys = Tensor.zeros(
            shape=self.cache_shape,
            dtype=self.params.dtype,
            device=device,
        )

        self.values = Tensor.zeros(
            shape=self.cache_shape, dtype=self.params.dtype, device=device
        )

        self.device = device

    @property
    def cache_shape(self) -> list[int]:
        return [
            self.max_seq_len,
            self.num_layers,
            self.max_cache_batch_size,
            self.params.n_kv_heads,
            self.params.head_dim,
        ]

    def fetch(
        self, seq_ids: list[int]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        existing_keys = list(self.cache_lengths.keys())
        for i, seq_id in enumerate(seq_ids):
            if existing_keys[i] != seq_id:
                msg = (
                    "seq_ids passed, are different than current inflight"
                    " batch.Naive Caching currently does not support mutating"
                    " inflight batches."
                )
                raise ValueError(msg)

        return (
            self.keys,
            self.values,
            Tensor.scalar(self.max_sequence_length, DType.int64, self.device),
            # TODO: MSDK-1201 - This next variable is not used upstream.
            # It is included here, as a placeholder, until we can dynamically
            # return a number of tensors from both `fetch` and `input_symbols`.
            Tensor.scalar(self.max_sequence_length, DType.int64, self.device),
        )

    def input_symbols(
        self,
    ) -> tuple[TensorType, TensorType, TensorType, TensorType]:
        return (
            # k_cache
            BufferType(
                self.params.dtype,
                shape=[
                    "max_seq_len",
                    "num_layers",
                    "batch_size",
                    "num_kv_heads",
                    "head_dim",
                ],
            ),
            # v_cache
            BufferType(
                self.params.dtype,
                shape=[
                    "max_seq_len",
                    "num_layers",
                    "batch_size",
                    "num_kv_heads",
                    "head_dim",
                ],
            ),
            # start_pos
            TensorType(DType.int64, shape=[]),
            # null_op - this isnt used for the naive cache
            TensorType(DType.int64, shape=[]),
        )
