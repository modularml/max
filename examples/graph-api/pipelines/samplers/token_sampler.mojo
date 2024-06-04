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

from tensor import Tensor
from utils import StaticTuple


@value
struct SamplerResult(Stringable):
    """Container for a token sampler decision. This struct retains some
    context on what options remained after filtering (to aid in
    rationalizing sampler behavior). The goal is to facilitate experimentation,
    not raw performance."""

    # Chosen token (vocabulary index)
    var selected: Int

    # Options the selected token was sampled from after filtering
    var options: List[Int]

    # List of the associated likelihoods (len(options) == len(likelihoods))
    var likelihoods: List[Float32]

    fn __init__(
        inout self: Self,
        selected: Int,
        options: List[Int] = List[Int](),
        likelihoods: List[Float32] = List[Float32](),
    ):
        self.selected = selected
        self.options = options
        self.likelihoods = likelihoods

    fn __str__(self) -> String:
        var msg = "Selected: " + str(self.selected) + " from "
        for i in range(len(self.options)):
            msg += (
                "["
                + str(self.options[i])
                + ", "
                + str(self.likelihoods[i])
                + "] "
            )

        return msg


trait TokenSampler:
    """A generic token sampler that takes in a list of logits and samples
    an element based on the associated likelihoods."""

    fn sample[dtype: DType](self, logits: Tensor[dtype]) -> SamplerResult:
        ...
