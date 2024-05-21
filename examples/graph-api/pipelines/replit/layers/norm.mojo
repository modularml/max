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

from max.graph import ops, TensorType, Symbol
from tensor import Tensor, TensorShape

from ..weights.hyperparams import HyperParams


@value
struct LPLayerNorm:
    """Low Precision Layer Normalization."""

    alias eps: Float32 = 1e-05
    var weight: Symbol
    var hyperparams: HyperParams

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        beta = g.constant(
            Tensor[DType.float32](TensorShape(self.hyperparams.d_model), 0)
        )
        out = ops.layer_norm(input, self.weight, beta, self.eps)
        return out
