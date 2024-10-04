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
"""Kernel Names for KV Cache related custom Ops."""

from enum import Enum
from max.dtype import DType
from max.driver import Device


class KVCacheLayout(Enum):
    BHSD = "bhsd"
    BSHD = "bshd"

    def __str__(self) -> str:
        return self.value


VALID_KV_KERNELS = [
    ("bf16", 1, 10, KVCacheLayout.BHSD),
    ("bf16", 1, 10, KVCacheLayout.BSHD),
    ("f32", 1, 10, KVCacheLayout.BHSD),
    ("f32", 1, 10, KVCacheLayout.BSHD),
    ("bf16", 8, 128, KVCacheLayout.BHSD),
    ("bf16", 8, 128, KVCacheLayout.BSHD),
    ("f32", 8, 128, KVCacheLayout.BHSD),
    ("f32", 8, 128, KVCacheLayout.BSHD),
    ("bf16", 8, 64, KVCacheLayout.BHSD),
    ("bf16", 8, 64, KVCacheLayout.BSHD),
    ("f32", 8, 64, KVCacheLayout.BHSD),
    ("f32", 8, 64, KVCacheLayout.BSHD),
]


class KVCacheType(Enum):
    CONTIGUOUS = "contiguous"
    CONTINUOUS = "continuous_batch"

    def __str__(self) -> str:
        return self.value


class KVCacheParams:
    def __init__(
        self,
        dtype: DType,
        n_kv_heads: int,
        head_dim: int,
        device: Device,
        cache_type: KVCacheType = KVCacheType.CONTIGUOUS,
    ):
        # Initialize static attributes.
        self.dtype = dtype
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.layout = (
            KVCacheLayout.BHSD if device.is_host else KVCacheLayout.BSHD
        )
        self.cache_type = cache_type

        # Validate inputs.
        if (
            self.dtype_shorthand,
            n_kv_heads,
            head_dim,
            self.layout,
        ) not in VALID_KV_KERNELS:
            raise ValueError(
                "Unsupported KV Cache Configuration: got dtype:"
                f" {self.dtype_shorthand}, n_kv_heads: {n_kv_heads}, head_dim:"
                f" {head_dim}, layout: {self.layout}"
            )

    @property
    def dtype_shorthand(self) -> str:
        """The textual representation in shorthand of the dtype."""
        return "bf16" if self.dtype == DType.bfloat16 else "f32"

    @property
    def static_cache_shape(self) -> tuple[str, str, str, str, str]:
        if self.layout == KVCacheLayout.BHSD:
            return (
                "num_layers",
                "batch_size",
                "n_kv_heads",
                "seq_len",
                "head_dim",
            )
        else:
            return (
                "num_layers",
                "batch_size",
                "seq_len",
                "n_kv_heads",
                "head_dim",
            )
