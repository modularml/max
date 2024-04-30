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

from extensibility import Tensor, empty_tensor
import python


@always_inline
fn numpy_data_pointer[
    type: DType
](numpy_array: PythonObject) raises -> DTypePointer[type]:
    return DTypePointer[type](
        address=int(numpy_array.__array_interface__["data"][0])
    )


@always_inline
fn memcpy_to_numpy[
    type: DType, rank: Int
](array: PythonObject, tensor: Tensor[type, rank]) raises:
    var dst = numpy_data_pointer[type](array)
    var src = tensor.data
    var length = tensor.nelems()
    memcpy(dst, src, length)


@always_inline
fn memcpy_from_numpy[
    type: DType, rank: Int
](array: PythonObject, tensor: Tensor[type, rank]) raises:
    var src = numpy_data_pointer[type](array)
    var dst = tensor.data
    var length = tensor.nelems()
    memcpy(dst, src, length)


@always_inline
fn shape_to_python_list[
    rank: Int
](shape: StaticIntTuple[rank]) raises -> PythonObject:
    var python_list = python.Python.evaluate("list()")
    for i in range(rank):
        _ = python_list.append(shape[i])
    return python_list^


@always_inline
fn get_np_dtype[type: DType](np: PythonObject) raises -> PythonObject:
    @parameter
    if type.is_float32():
        return np.float32
    elif type.is_int32():
        return np.int32
    elif type.is_int64():
        return np.int64
    elif type.is_uint8():
        return np.uint8

    raise "Unknown datatype"


@always_inline
fn tensor_to_numpy[
    type: DType
](tensor: Tensor[type], np: PythonObject) raises -> PythonObject:
    var shape = shape_to_python_list(tensor.shape)
    var tensor_as_numpy = np.zeros(shape, get_np_dtype[type](np))
    _ = shape^
    memcpy_to_numpy(tensor_as_numpy, tensor)
    return tensor_as_numpy^


@always_inline
fn numpy_to_tensor[
    type: DType, rank: Int
](array: PythonObject) raises -> Tensor[type, rank]:
    var shape = StaticIntTuple[rank]()
    var array_shape = array.shape
    for i in range(rank):
        shape[i] = array_shape[i].__index__()
    var out = empty_tensor[type, rank]((shape))
    memcpy_from_numpy(array, out)
    return out^
