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

from max.tensor import Tensor
import max.driver as driver


@value
struct SamplerResult(Stringable):
    """Container for a token sampler decision. This struct retains some
    context on what options remained after filtering (to aid in
    rationalizing sampler behavior). The goal is to facilitate experimentation,
    not raw performance."""

    var selected: Int
    """Chosen token (vocabulary index)."""

    var options: List[Int]
    """Options the selected token was sampled from after filtering."""

    var likelihoods: List[Float32]
    """List of the associated likelihoods (len(options) == len(likelihoods))."""

    fn __init__(
        out self,
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

    def sample[
        dtype: DType, rank: Int
    ](self, owned logits: driver.Tensor[dtype, rank]) -> SamplerResult:
        ...
