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


from random import randn
from python import Python
from .python_utils import tensor_to_numpy, numpy_to_tensor
from max import register
from max.extensibility import Tensor, empty_tensor


@register.op("monnx.det_v11")
fn det[type: DType, rank: Int](x: Tensor[type, rank]) -> Tensor[type, rank - 2]:
    try:
        var np = Python.import_module("numpy")
        var np_array = tensor_to_numpy(x, np)
        var np_out = np.linalg.det(np_array)
        return numpy_to_tensor[type, rank - 2](np_out)
    except e:
        print(e)
    return empty_tensor[type, rank - 2](0)
