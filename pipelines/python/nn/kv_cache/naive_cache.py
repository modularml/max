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

from max.driver import Device, Tensor
from max.dtype import DType
import numpy as np


class NaiveKVCache:
    keys: Tensor
    values: Tensor
    sequence_length: int

    def __init__(
        self,
        max_length: int,
        max_batch_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: Device,
    ):
        self.keys = Tensor.zeros(
            shape=(max_length, n_layers, max_batch_size, n_kv_heads, head_dim),
            dtype=DType.float32,
            device=device,
        )
        self.values = Tensor.zeros(
            shape=(max_length, n_layers, max_batch_size, n_kv_heads, head_dim),
            dtype=DType.float32,
            device=device,
        )
        self.sequence_length = 0
