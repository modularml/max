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

from max.tensor import Tensor, TensorShape
from max.graph import ops, Dim, Symbol, TensorType, _OpaqueType as OpaqueType
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from kv_cache.types import ContiguousKVCache, KVCacheStaticParams
from max.graph.kv_cache.kernel_names import _kv_cache_kernel_names

from pipelines.nn import Linear
from pipelines.nn.attention import rope


@value
struct KVCacheOptimizedAttention[type: DType, kv_params: KVCacheStaticParams]:
    """Attention block that supports the specialized ContiguousKVCache type."""

    alias _kernel_names = _kv_cache_kernel_names[type, kv_params]()

    # hyperparams
    var n_heads: Int
    var dim: Int

    # weights/projections
    var wqkv: Symbol
    var wo: Linear

    # scalar index of the current layer. Used to retrieve the kv cache objects inside of each kernel.
    var layer_idx: Symbol

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        freqs_cis: Symbol,
        kv_collection: Symbol,
        mask: Symbol,
        valid_lengths: Symbol,
    ) -> Tuple[Symbol, Symbol]:
        """Constructs the forward pass for this attention block

        input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
        start_pos: Scalar with index of starting token, effectively tracks
            the number of entries in the cache.
        freqs_cis: Positional frequencies tensor with shape
            (seq_len, head_dim // 2, 2).
        """

        g = input.graph()

        # extract shape characteristics of the input
        batch_size, seq_len = input.shape()[0], input.shape()[1]

        # reshape our rope positional frequencies
        f_shape = ops.shape_of(freqs_cis)
        new_f_shape = ops.stack(List[Symbol](f_shape[0], g.scalar(Int64(-1))))
        freqs_cis_2d = ops.reshape(freqs_cis, new_f_shape)

        xq_type = input.type()

        xq = ops.custom[self._kernel_names.fused_qkv_matmul_kernel](
            List[Symbol](input, self.wqkv, kv_collection, self.layer_idx),
            xq_type,
        )

        xq = xq.reshape(
            batch_size, seq_len, self.n_heads, int(kv_params.head_size)
        )
        xq = ops.custom[self._kernel_names.fused_qk_rope_kernel](
            List[Symbol](
                xq, kv_collection, freqs_cis_2d, self.layer_idx, g.scalar(False)
            ),
            xq.type(),
        )

        # do flash attention
        seq_len_sym = ops.shape_of(input)[1]
        var attn_mask = attention_mask[type](mask, start_pos, seq_len_sym, type)
        var output_type = xq.type()
        attn_out = ops.custom[self._kernel_names.flash_attention_kernel](
            List[Symbol](
                xq,
                kv_collection,
                self.layer_idx,
                attn_mask,
                valid_lengths,
                g.scalar(isqrt(Float32(kv_params.head_size))),
            ),
            output_type,
        )

        attn_out = attn_out.reshape(batch_size, seq_len, -1)

        # final projection and return
        return attn_out @ self.wo, kv_collection


def attention_mask[
    type: DType
](
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
                g.scalar(-10000, activation_dtype),
                ops.stack(List[Symbol](seq_len, seq_len)),
            ),
            TensorType(
                activation_dtype,
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

    y = g.full[type](-10000.0, x.shape())

    return ops.select(select_mask, x, y)
