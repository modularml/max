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
from max.tensor import Tensor, TensorShape

from ..weights.hyperparams import HyperParams


@value
struct LPLayerNorm[dtype: DType]:
    """Low Precision Layer Normalization.

    Parameters:
        dtype: The DType of the weights and inputs to this layer.
    """

    alias eps: Scalar[dtype] = 1e-05
    var weight: Symbol
    var hyperparams: HyperParams

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        with g.layer("LPlayerNorm"):
            beta = g.constant(
                Tensor[dtype](TensorShape(self.hyperparams.d_model), 0)
            )
            # Since norm weights are float32, cast to input dtype to avoid
            # promoting the result to float32 when the input is float16.
            out = ops.layer_norm(
                input,
                ops.cast(self.weight, input.tensor_type().dtype),
                beta,
                self.eps,
            )
            return out
