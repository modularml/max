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

from max.graph import TensorValue, ValueLike, Weight, ops

from .layer import Layer


@dataclass
class Linear(Layer):
    """A fully connected layer."""

    weight: ValueLike

    def __call__(self, x: TensorValue) -> TensorValue:
        weight = TensorValue(self.weight)
        if (
            isinstance(self.weight, Weight)
            and self.weight.quantization_encoding is not None
        ):
            return ops.qmatmul(self.weight.quantization_encoding, x, weight)
        return x @ weight.T


@dataclass
class MLP(Layer):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear

    def __call__(self, x: ValueLike) -> TensorValue:
        return self.down_proj((ops.silu(self.gate_proj(x)) * self.up_proj(x)))
