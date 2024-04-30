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

from max.engine import InferenceSession
from max.graph import Graph, TensorType, Type, ops
from tensor import Tensor, TensorShape, randn
from pathlib import Path


# This example highlights the very basic API structure around building a MAX
# Graph model and executing it through the MAX engine APIS.
# Simply run this mojo file to create, load, and execute this simple model.
fn construct_graph[op_name: StringLiteral]() raises -> Graph:
    var graph = Graph("BasicModel", List[Type](TensorType(DType.float32, 2, 6)))

    # Create a constant for usage in the matmul op below:
    var matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
    var matmul_constant = graph.constant(matmul_constant_value)

    # Start adding a sequence of operator calls to build the graph:
    # NOTE: The first accessor on graph is the first input.
    var matmul = graph[0] @ matmul_constant
    var gelu = ops.custom[op_name](matmul, matmul.type())
    graph.output(gelu)
    return graph


fn main() raises:
    var session = InferenceSession()
    var model1 = session.load(
        construct_graph["my_gelu"](),
        custom_ops_paths=Path("custom_ops.mojopkg"),
    )
    var model2 = session.load(
        construct_graph["my_tanh_gelu"](),
        custom_ops_paths=Path("custom_ops.mojopkg"),
    )
    var model3 = session.load(
        construct_graph["my_sigmoid_gelu"](),
        custom_ops_paths=Path("custom_ops.mojopkg"),
    )

    # Create some sample input to run through the model:
    var input = randn[DType.float32]((2, 6))

    var results1 = model1.execute("input0", input)
    var output1 = results1.get[DType.float32]("output0")
    print("my_gelu", output1)

    var results2 = model2.execute("input0", input)
    var output2 = results2.get[DType.float32]("output0")
    print("my_tanh_gelu", output2)

    var results3 = model3.execute("input0", input)
    var output3 = results3.get[DType.float32]("output0")
    print("my_sigmoid_gelu", output3)
