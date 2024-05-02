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


from max.extensibility import Tensor, empty_tensor
from max import register
from math import erf, exp, sqrt, tanh
from random import randn


@register.op("my_gelu")
fn gelu[type: DType, rank: Int](x: Tensor[type, rank]) -> Tensor[type, rank]:
    var output = empty_tensor[type](x.shape)

    @always_inline
    @parameter
    fn func[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
        var tmp = x.simd_load[width](i)
        return tmp / 2 * (1 + erf(tmp / sqrt(2)))

    print("Hello, custom GELU!")
    output.for_each[func]()
    return output^


@register.op("my_tanh_gelu")
fn gelu_tanh_approx[
    type: DType, rank: Int
](x: Tensor[type, rank]) -> Tensor[type, rank]:
    var output = empty_tensor[type](x.shape)

    @always_inline
    @parameter
    fn func[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
        var tmp = x.simd_load[width](i)
        return (
            0.5 * tmp * (1 + tanh(0.7978845608 * (tmp + 0.044715 * tmp**3)))
        )

    print("Hello, custom tanh GELU!")
    output.for_each[func]()
    return output^


@register.op("my_sigmoid_gelu")
fn gelu_sigmoid_approx[
    type: DType, rank: Int
](x: Tensor[type, rank]) -> Tensor[type, rank]:
    var output = empty_tensor[type](x.shape)

    @always_inline
    @parameter
    fn func[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
        var tmp = x.simd_load[width](i)
        return tmp * (1 / (1 + exp(-1.702 * tmp)))

    print("Hello, custom sigmoid GELU!")
    output.for_each[func]()
    return output^
