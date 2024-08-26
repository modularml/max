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
from math import isqrt

import mo
from max.tensor import Tensor, TensorShape
from max.graph import ops, Dim, Symbol, TensorType, _OpaqueType as OpaqueType
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from kv_cache.types import (
    ContiguousKVCache,
    KVCacheLayout,
    KVCacheStaticParams,
    KVCacheKernelNames,
)
from max.serve.kv_cache.kernel_names import _kv_cache_kernel_names

from pipelines.nn import Linear
from pipelines.nn.attention import rope


@value
struct KVCacheOptimizedAttention[kv_params: KVCacheStaticParams]:
    """Attention block that supports the specialized ContiguousKVCache type."""

    alias _kernel_names = _kv_cache_kernel_names[DType.float32, kv_params]()

    # hyperparams
    var n_heads: Int
    var dim: Int

    # weights/projections
    var wqkv: Symbol
    var wo: Linear

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        freqs_cis: Symbol,
        k_cache: Symbol,
        v_cache: Symbol,
        mask: Symbol,
    ) -> Tuple[Symbol, Symbol, Symbol]:
        """Constructs the forward pass for this attention block

        input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
        start_pos: Scalar with index of starting token, effectively tracks
            the number of entries in the cache.
        freqs_cis: Positional frequencies tensor with shape
            (seq_len, head_dim // 2, 2).
        k_cache: Previously computed keys. This is a mo.opaque ContiguousKVCache object
            with logical shape (batch, prev_seq_len, n_kv_heads, head_dim).
        v_cache: Previously computed values. This is a mo.opaque ContiguousKVCache object
            with logical shape (batch, prev_seq_len, n_kv_heads, head_dim).
        """

        g = input.graph()

        # extract shape characteristics of the input
        batch_size, seq_len = input.shape()[0], input.shape()[1]
        head_dim = g.scalar[DType.float32](kv_params.head_size)

        # define opaque types for custom op outputs
        # TODO give these guys actual values for num_kv_head and head_size
        # We only use these types to get `id()`, and the actual value of this
        # string is not used.
        var k_cache_type = OpaqueType(
            ContiguousKVCache[DType.float32, kv_params].id()
        )
        var v_cache_type = OpaqueType(
            ContiguousKVCache[DType.float32, kv_params].id()
        )

        # reshape our rope positional frequencies
        f_shape = ops.shape_of(freqs_cis)
        new_f_shape = ops.stack(List[Symbol](f_shape[0], g.scalar(Int64(-1))))
        freqs_cis_2d = ops.reshape(freqs_cis, new_f_shape)

        xq_type = input.type()

        xq = ops.custom[self._kernel_names.fused_qkv_matmul_kernel](
            List[Symbol](input, self.wqkv, k_cache, v_cache),
            xq_type,
        )

        xq = xq.reshape(batch_size, seq_len, self.n_heads, kv_params.head_size)
        xq = ops.custom[self._kernel_names.fused_qk_rope_kernel](
            List[Symbol](xq, k_cache, freqs_cis_2d), xq.type()
        )

        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            # Flash Attention shapes differ on CPU and GPU, we need to
            # transpose on cpu. This'll will be fixed by KERN-626
            xq = xq.swapaxes(1, 2)

        # do flash attention
        seq_len_sym = ops.shape_of(input)[1]
        var attn_mask = attention_mask(
            mask, start_pos, seq_len_sym, DType.float32
        )
        var output_type = xq.type()
        attn_out = ops.custom[self._kernel_names.flash_attention_kernel](
            List[Symbol](xq, k_cache, v_cache, attn_mask, ops.rsqrt(head_dim)),
            output_type,
        )

        # transpose hidden state to (batch_size, seq_len, num_heads * head_dim)
        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            # Flash Attention shapes differ on CPU and GPU, we need to
            # transpose on cpu. This'll will be fixed by KERN-626
            attn_out = attn_out.swapaxes(1, 2)
        attn_out = attn_out.reshape(batch_size, seq_len, -1)

        # final projection and return
        return attn_out @ self.wo, k_cache, v_cache


def attention_mask(
    mask: Symbol, start_pos: Symbol, seq_len: Symbol, activation_dtype: DType
) -> Symbol:
    g = start_pos.graph()

    seq_len = seq_len.reshape()
    start_pos = start_pos.reshape()

    # Mask out current sequence elements [i, j] where j > i with an
    # upper-triangular matrix filled with -inf.
    mask_val = ops.cast(
        g.op(
            "rmo.mo.broadcast_to",
            List(
                g.scalar(-10000, DType.float32),
                ops.stack(List[Symbol](seq_len, seq_len)),
            ),
            TensorType(
                DType.float32,
                "seq_len",
                "seq_len",
            ),
        ),
        activation_dtype,
    )
    new_mask = ops.band_part(
        mask_val,
        g.scalar[DType.int64](-1),
        num_upper=g.scalar[DType.int64](0),
        # Invert the mask from lower to upper.
        exclude=True,
    )

    zeros = g.op(
        "rmo.mo.broadcast_to",
        List(
            g.scalar(0, activation_dtype),
            ops.stack(List[Symbol](seq_len, start_pos)),
        ),
        TensorType(
            activation_dtype,
            "seq_len",
            "start_pos",
        ),
    )

    full_seq_len = Dim("full_seq_len")
    x = ops.concat(
        List[Symbol](
            zeros,
            new_mask,
        ),
        axis=1,
        out_dim=full_seq_len,
    )

    # In the above, x, results in a seq_len/start_pos + seq_len tensor
    # x, has 0s with the upper-triangular mapped to -inf
    # to accomodate for left padding, we should create a new tensor of -inf
    # with the same shape of x, and return the values of this new -inf tensor
    # when a padded token is present and x when a valid token is present
    select_mask = g.op(
        "rmo.mo.broadcast_to",
        List(mask, ops.stack(List[Symbol](seq_len, ops.shape_of(x)[1]))),
        TensorType(DType.bool, "seq_len", full_seq_len),
    )

    y = g.full[DType.float32](-10000.0, x.shape())

    return ops.select(select_mask, x, y)
