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

from dataclasses import dataclass
from typing import Tuple, Union

from max.graph import TensorValue, TensorValueLike, Weight, dtype_promotion, ops
from .layer import Layer


@dataclass
class Conv2D(Layer):
    """A 2D convolution over an input signal composed of several input
    planes.
    """

    filter: TensorValueLike

    stride: Union[int, Tuple[int, int]] = (1, 1)
    padding: Union[int, Tuple[int, int, int, int]] = (0, 0, 0, 0)
    dilation: Union[int, Tuple[int, int]] = (1, 1)
    groups: int = 1
    bias: bool = False

    def __call__(self, x: TensorValue) -> TensorValue:
        filter = dtype_promotion._promote_to(self.filter, x.dtype)
        # filter = ops.constant(self.filter, x.dtype)

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            return ops.conv2d(
                x,
                filter.quantization_encoding,
                self.stride,
                self.dilation,
                self.padding,
                self.groups,
            )
        return ops.conv2d(
            x, filter, self.stride, self.dilation, self.padding, self.groups
        )
