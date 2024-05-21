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

from pathlib import Path

from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
from tensor import Tensor, TensorShape


def construct_graph[op_name: StringLiteral]() -> Graph:
    graph = Graph(TensorType(DType.float32, 2, 6))

    matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
    matmul_constant = graph.constant(matmul_constant_value)

    matmul = graph[0] @ matmul_constant
    gelu = ops.custom[op_name](matmul, matmul.type())
    graph.output(gelu)
    return graph


def main():
    # Load the graph with custom ops package
    session = InferenceSession()
    # Try changing the op_name to a different op from gelu.mojo
    model = session.load(
        construct_graph["my_gelu"](),
        custom_ops_paths=Path("custom_ops.mojopkg"),
    )

    # Create some sample input to run through the model:
    input = Tensor[DType.float32].randn(TensorShape(2, 6))
    results = model.execute("input0", input)
    output = results.get[DType.float32]("output0")
    print(output)
