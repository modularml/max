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

from .token_sampler import TokenSampler, SamplerResult
from tensor import Tensor
from random import random_float64
import math
from utils.numerics import min_finite


@value
struct WeightedSampler(TokenSampler):
    # Standard temperature parameter -- 1.0 is unmodified, 0.0 is effectively greedy sampling
    var temperature: Float32
    # min_p style filter (source: https://github.com/ggerganov/llama.cpp/pull/3841)
    var min_p: Float32

    fn __init__(inout self: Self, temperature: Float32, min_p: Float32 = 0.05):
        self.temperature = temperature
        self.min_p = min_p

    fn sample[dtype: DType](self, logits: Tensor[dtype]) -> SamplerResult:
        var normalization = Scalar[DType.float32](0)

        # Add a floor to mitigate div0 if T=0.0 is passed in.
        var temp_modified: SIMD[DType.float32, 1] = max(
            Float32(1e-6), self.temperature
        )

        # Overflow mitigation.
        # p_i = exp(logit_i / T) / (sum_j exp(logit_j / T))
        #     = exp(logit_max / T) / exp(logit_max / T) (...)
        #     = exp((logit_i-logit_max)/T) / (sum_j exp((logit_j-logit_max)/T))
        var largest = min_finite[dtype]()

        for i in range(logits.num_elements()):
            if largest < logits[0, i]:
                largest = logits[0, i]

        for i in range(logits.num_elements()):
            var intermediate: SIMD[DType.float32, 1] = (
                logits[0, i] - largest
            ).cast[DType.float32]() / temp_modified
            var p = math.exp(intermediate)
            normalization += p

        # Start filtering for min_p
        var retained_idx = List[Int]()
        var retained_p = List[Float32]()
        var options = List[Int]()
        var likelihoods = List[Float32]()

        # Now run through again with the actual probabilities
        for i in range(logits.num_elements()):
            var intermediate: SIMD[DType.float32, 1] = (
                logits[0, i] - largest
            ).cast[DType.float32]() / temp_modified
            var p: Float32 = math.exp(intermediate) / normalization

            if p >= (self.min_p / normalization):
                retained_idx.append(i)
                retained_p.append(p)

        # Renormalize after filtering min_p
        normalization = Scalar[DType.float32](0)
        for v in range(len(retained_idx)):
            normalization += retained_p[v]

        # Simple O(N) weighted sampler
        # Collect the considered tokens as we go for the SamplerResult
        var u = random_float64()
        var cdf = Scalar[dtype.float32](0.0)
        for i in range(len(retained_idx)):
            options.append(retained_idx[i])
            likelihoods.append(
                retained_p[i] / normalization.cast[DType.float32]()
            )

            cdf += retained_p[i] / normalization

            if cdf > u:
                return SamplerResult(retained_idx[i], options)

        return SamplerResult(retained_idx[len(retained_idx) - 1], options)
