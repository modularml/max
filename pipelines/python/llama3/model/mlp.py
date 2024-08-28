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

"""Multi-layer Perceptron."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from max.graph import TensorValue, ops, ValueLike
from max.graph.quantization import QuantizationEncoding


@dataclass
class Linear:
    """A fully connected layer."""

    weight: TensorValue
    quantization_encoding: Optional[QuantizationEncoding] = None

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.quantization_encoding is not None:
            return ops.qmatmul(self.quantization_encoding, x, self.weight)
        return x @ self.weight


@dataclass
class MLP:
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear

    def __call__(self, x: ValueLike) -> TensorValue:
        return self.down_proj((ops.silu(self.gate_proj(x)) * self.up_proj(x)))
