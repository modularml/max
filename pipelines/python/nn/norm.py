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

"""Normalization layer."""

from dataclasses import dataclass

from max.graph import TensorValue, ops, ValueLike


@dataclass
class RMSNorm:
    weight: ValueLike
    eps: float = 1e-6

    def __call__(self, x: ValueLike) -> TensorValue:
        scale = ops.rsqrt(ops.mean(x**2, axis=-1) + self.eps)

        # Cast back to the activation dtype, which may differ from the weights.
        #
        # Checkpoints can store weights in a precision that differs from our
        # desired activation precision.
        # For example we may want computations to be in bfloat16, but the
        # weights or a subset thereof are stored as float32.
        # So to prevent dtype promotion causing issues downstream, just cast
        # to the activation dtype here.
        return x * scale * ops.cast(self.weight, x.dtype)
