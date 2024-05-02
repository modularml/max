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

from max import engine
import numpy as np

session = engine.InferenceSession()
model = session.load("onnx_det.onnx", custom_ops_path="custom_ops.mojopkg")

for tensor in model.input_metadata:
    print(f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}")

input_x = np.random.rand(3, 3, 5).astype(np.float32)
input_a = np.random.rand(5, 3).astype(np.float32)
input_b = np.random.rand(3).astype(np.float32)

result = model.execute(X=input_x, A=input_a, B=input_b)
print(result)
