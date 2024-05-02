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

# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #


from max.extensibility import Tensor, empty_tensor
from max import register

from algorithm.functional import vectorize, sync_parallelize
from buffer.list import Dim, DimList
from math import min, div_ceil
from runtime.llcl import Runtime


@register.op("ggml_rope")
fn ggml_rope[
    inType: DType, freqType: DType
](input: Tensor[inType, 4], freqs: Tensor[freqType, 2]) -> Tensor[inType, 4]:
    """
    Implements the RoPE Kernel:
    https://github.com/meta-llama/llama/blob/b8348da38fde8644ef00a56596efb376f86838d1/llama/model.py#L132
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).

    Args:
        input: Input tensor of shape (batch, seq_len, n_heads, head_dim).
        freqs: Freqs tensor of shape (seq_len, head_dim).


    This is a bit like a broadcasted 4D complex mul operation:
    input[A,B,C,D] * freqs[1,B,1,D]
    """

    constrained[input.type.is_floating_point(), "input must be float"]()
    constrained[freqs.type.is_floating_point(), "freqs must be float"]()

    var output = empty_tensor[input.type](input.shape)

    var n_batch = input.shape[0]
    var n_tok = input.shape[1]
    var n_heads = input.shape[2]
    var n_head_dim = input.shape[3]
    var static_n_head_dim = input.shape[3]

    # Parallelize on outer dimensions.
    var outer_len = n_batch * n_tok * n_heads
    # TODO: find a heuristic to replace this number.
    var MIN_TASK_SIZE = 32768
    var num_workers = min(
        Runtime().parallelism_level(),
        div_ceil(outer_len * n_head_dim, MIN_TASK_SIZE),
    )
    var chunk_size = div_ceil(outer_len, num_workers)

    @__copy_capture(n_tok, n_heads, n_head_dim, chunk_size)
    @parameter
    fn task_func(thread_id: Int):
        var start_idx = thread_id * chunk_size
        var end_idx = min((thread_id + 1) * chunk_size, outer_len)
        for i in range(start_idx, end_idx):
            var i0 = i // (n_tok * n_heads)
            var i1 = (i // n_heads) % n_tok
            var i2 = i % n_heads

            @parameter
            @always_inline
            fn func[width: Int](i3: Int):
                var idx = StaticIntTuple[4](i0, i1, i2, i3 * 2)
                var f_idx = StaticIntTuple[2](i1, i3 * 2)

                var x_c = input.simd_load[width * 2](idx).deinterleave()
                var x_re = x_c[0]
                var x_im = x_c[1]

                var f_c = freqs.simd_load[width * 2](f_idx).cast[
                    inType
                ]().deinterleave()
                var f_re = f_c[0]
                var f_im = f_c[1]

                var r_re = (x_re * f_re) - (x_im * f_im)
                var r_im = (x_re * f_im) + (x_im * f_re)

                var r_c = r_re.interleave(r_im)
                output.store(idx, r_c)

            # Vectorize on n_head_dim // 2
            #  (the outputs needs to be computed 2 at a time).
            alias simd_width = simdwidthof[input.type]()

            vectorize[func, simd_width](n_head_dim // 2)

    # Manually parallelize / vectorize the loop since output.for_each
    # isn't adapted to complex numbers.
    sync_parallelize[task_func](num_workers)
    return output^
