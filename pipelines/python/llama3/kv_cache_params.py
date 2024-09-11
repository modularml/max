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
]


class KVCacheParams:
    def __init__(
        self,
        dtype: DType,
        n_kv_heads: int,
        head_dim: int,
        device: Device,
    ):
        # Initialize static attributes
        self.dtype = dtype
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.layout = (
            KVCacheLayout.BHSD if device.is_host else KVCacheLayout.BSHD
        )

        # Validate inputs
        dt = "bf16" if dtype == DType.bfloat16 else "f32"
        if (dt, n_kv_heads, head_dim, self.layout) not in VALID_KV_KERNELS:
            raise Exception(
                f"Unsupported KV Cache Configuration: got dtype: {dt},"
                f" n_kv_heads: {n_kv_heads}, head_dim: {head_dim}, layout:"
                f" {self.layout}"
            )

        # Create Kernel Names for Ease
        self._matmul_kernel = (
            f"matmul_kv_cache_h{n_kv_heads}_d{head_dim}_{self.layout}"
        )
        self._flash_attention_kernel = (
            f"flash_attention_kv_cache_h{n_kv_heads}_d{head_dim}_{self.layout}"
        )
        self._kv_cache_length_kernel = (
            f"kv_cache_length_h{n_kv_heads}_d{head_dim}_{self.layout}_{dt}"
        )
        self._key_cache_for_layer_kernel = (
            f"key_cache_for_layer_h{n_kv_heads}_d{head_dim}_{self.layout}_{dt}"
        )
        self._value_cache_for_layer_kernel = f"value_cache_for_layer_h{n_kv_heads}_d{head_dim}_{self.layout}_{dt}"
        self._fused_qkv_matmul_kernel = (
            f"fused_qkv_matmul_kv_cache_h{n_kv_heads}_d{head_dim}_{self.layout}"
        )
        self._fused_qk_rope_kernel = (
            f"fused_qk_rope_h{n_kv_heads}_d{head_dim}_{self.layout}"
        )

    @property
    def matmul_kernel(self) -> str:
        return self._matmul_kernel

    @property
    def flash_attention_kernel(self) -> str:
        return self._flash_attention_kernel

    @property
    def kv_cache_length_kernel(self) -> str:
        return self._kv_cache_length_kernel

    @property
    def key_cache_for_layer_kernel(self) -> str:
        return self._key_cache_for_layer_kernel

    @property
    def value_cache_for_layer_kernel(self) -> str:
        return self._value_cache_for_layer_kernel

    @property
    def fused_qkv_matmul_kernel(self) -> str:
        return self._fused_qkv_matmul_kernel

    @property
    def fused_qk_rope_kernel(self) -> str:
        return self._fused_qk_rope_kernel
