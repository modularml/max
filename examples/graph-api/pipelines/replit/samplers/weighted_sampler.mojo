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

from sys import simdwidthof
from .token_sampler import TokenSampler, SamplerResult
from max.tensor import Tensor
from random import random_float64
import math
from memory import UnsafePointer
from utils.numerics import min_finite
import max.driver as driver
from runtime.tracing import Trace, TraceLevel


@value
struct WeightedSampler(TokenSampler):
    """A random token sampler.
    Source: https://github.com/ggerganov/llama.cpp/pull/3841.
    """

    var temperature: Float32
    """Standard temperature parameter -- 1.0 is unmodified, 0.0 is effectively
    greedy sampling."""

    var min_p: Float32
    """Minimum required starting percentage for sampled tokens."""

    def __init__(out self, temperature: Float32, min_p: Float32 = 0.05):
        self.temperature = temperature
        self.min_p = min_p

    def sample[dtype: DType](self, logits: Tensor[dtype]) -> SamplerResult:
        """Generates a random sample from the logits.

        Args:
          logits: Tensor logits. Shape must be [1, vocab_size].

        Returns:
          A SamplerResult with the selected token.
        """
        with Trace[TraceLevel.OP]("PipelineMetric.weighted_sampler"):
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

            var num_logits = logits.num_elements()

            for i in range(num_logits):
                if largest < logits[0, i]:
                    largest = logits[0, i]

            for i in range(num_logits):
                var intermediate: SIMD[DType.float32, 1] = (
                    logits[0, i] - largest
                ).cast[DType.float32]() / temp_modified
                var p = math.exp(intermediate)
                normalization += p

            # Start filtering for min_p
            var retained_idx = List[Int](capacity=num_logits)
            var retained_p = List[Float32](capacity=num_logits)

            # Now run through again with the actual probabilities
            for i in range(num_logits):
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

            var options = List[Int](capacity=len(retained_idx))
            var likelihoods = List[Float32](capacity=len(retained_idx))

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

                if cdf.cast[DType.float64]() > u:
                    return SamplerResult(retained_idx[i], options)

            return SamplerResult(retained_idx[len(retained_idx) - 1], options)

    # TODO (MSDK-491): Unify these and delete the other.
    def _sample[
        dtype: DType, rank: Int
    ](self, owned logits: driver.Tensor[dtype, rank]) -> SamplerResult:
        """Generates a random sample from the logits.

        Args:
          logits: Tensor logits. Shape must be [1, vocab_size].

        Returns:
          A SamplerResult with the selected token.
        """
        with Trace[TraceLevel.OP]("PipelineMetric.weighted_sampler"):
            # Add a floor to mitigate div0 if T=0.0 is passed in.
            var temp_modified: Float32 = max(Float32(1e-6), self.temperature)

            var length = logits.spec().num_elements()

            alias simd_width = simdwidthof[dtype]()
            var aligned_length = length & ~(simd_width - 1)

            var p_buf = UnsafePointer[Float32].alloc(length)

            # Overflow mitigation.
            # p_i = exp(logit_i / T) / (sum_j exp(logit_j / T))
            #     = exp(logit_max / T) / exp(logit_max / T) (...)
            #     = exp((logit_i-logit_max)/T) / (sum_j exp((logit_j-logit_max)/T))
            @always_inline
            @parameter
            fn reduce_max[simd_width: Int](start: Int, end: Int) -> Float32:
                var largest = SIMD[DType.float32, simd_width].MIN_FINITE
                var logits_ptr = logits.unsafe_ptr()

                for i in range(start, end, simd_width):
                    var v = (logits_ptr + i).load[width=simd_width]().cast[
                        DType.float32
                    ]()
                    (p_buf + i).store(v)
                    largest = max(largest, v)

                return largest.reduce_max()

            var largest = max(
                reduce_max[simd_width](0, aligned_length),
                reduce_max[1](aligned_length, length),
            )

            _ = logits^

            @always_inline
            @parameter
            fn exp_and_accumulate[
                simd_width: Int
            ](start: Int, end: Int) -> Float32:
                var normalization = SIMD[DType.float32, simd_width](0)

                for i in range(start, end, simd_width):
                    var intermediate = (
                        (p_buf + i).load[width=simd_width]() - largest
                    ) / temp_modified
                    var p = math.exp(intermediate)
                    (p_buf + i).store(p)
                    normalization += p

                return normalization.reduce_add()

            var normalization = exp_and_accumulate[simd_width](
                0, aligned_length
            ) + exp_and_accumulate[1](aligned_length, length)

            # Start filtering for min_p
            var retained_idx = List[Int](capacity=length)
            var retained_p = List[Float32](capacity=length)

            # Now run through again with the actual probabilities
            for i in range(length):
                var p = p_buf[i] / normalization

                if p >= (self.min_p / normalization):
                    retained_idx.append(i)
                    retained_p.append(p)

            p_buf.free()

            # Renormalize after filtering min_p
            normalization = Scalar[DType.float32](0)
            for v in range(len(retained_idx)):
                normalization += retained_p[v]

            var options = List[Int](capacity=len(retained_idx))
            var likelihoods = List[Float32](capacity=len(retained_idx))

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

                if cdf.cast[DType.float64]() > u:
                    return SamplerResult(retained_idx[i], options)

            return SamplerResult(retained_idx[len(retained_idx) - 1], options)
