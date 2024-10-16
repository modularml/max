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


VALID_KV_KERNELS = [
    ("bf16", 1, 16),
    ("f32", 1, 16),
    ("bf16", 8, 128),
    ("f32", 8, 128),
    ("bf16", 8, 64),
    ("f32", 8, 64),
]


class KVCacheStrategy(str, Enum):
    NAIVE = "naive"
    CONTINUOUS = "continuous"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def uses_opaque(self) -> bool:
        return self != KVCacheStrategy.NAIVE


class KVCacheParams:
    def __init__(
        self,
        dtype: DType,
        n_kv_heads: int,
        head_dim: int,
        cache_strategy: KVCacheStrategy = KVCacheStrategy.CONTINUOUS,
    ):
        # Initialize static attributes.
        self.dtype = dtype
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache_strategy = cache_strategy

        # Validate inputs.
        if (
            self.dtype_shorthand,
            n_kv_heads,
            head_dim,
        ) not in VALID_KV_KERNELS:
            raise ValueError(
                "Unsupported KV Cache Configuration: got dtype:"
                f" {self.dtype_shorthand}, n_kv_heads: {n_kv_heads}, head_dim:"
                f" {head_dim}"
            )

    @property
    def dtype_shorthand(self) -> str:
        """The textual representation in shorthand of the dtype."""
        return "bf16" if self.dtype == DType.bfloat16 else "f32"

    @property
    def static_cache_shape(self) -> tuple[str, str, str, str, str]:
        return (
            "num_layers",
            "batch_size",
            "seq_len",
            "n_kv_heads",
            "head_dim",
        )
