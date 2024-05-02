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

from onnx import TensorProto, OperatorSetIdProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
)
from onnx.checker import check_model

X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 3, 5])
A = make_tensor_value_info("A", TensorProto.FLOAT, [5, 3])
B = make_tensor_value_info("B", TensorProto.FLOAT, [3])
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])

node1 = make_node("MatMul", ["X", "A"], ["XA"])
node2 = make_node("Add", ["XA", "B"], ["Y"])
node3 = make_node("Det", ["Y"], ["Z"])

graph = make_graph(
    [node1, node2, node3], "lr", [X, A, B], [Z]  # nodes  # a name  # inputs
)  # outputs

onnx_model = make_model(graph, opset_imports=[OperatorSetIdProto(version=18)])
check_model(onnx_model)
with open("onnx_det.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
