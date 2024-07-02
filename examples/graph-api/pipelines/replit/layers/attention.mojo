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
"""The attention mechanism used within the model."""

import math
from collections import Optional
from utils.numerics import min_finite

from max.graph import ops, Dim, TensorType, Symbol
from max.graph.error import error
from max.tensor import Tensor, TensorShape

from ..weights.hyperparams import HyperParams
from ..layers.linear import Linear


@always_inline
def tril(input: Symbol, k: Int = 0) -> Symbol:
    """Gets the bottom triangle of the input.

    The upper triangle above the kth diagnoal is zero'd out.

    Args:
        input: The input tensor.
        k: The diagonal at which values at and below are True, and values
          above are False.

    Returns:
        A Dtype.bool matrix.

    Raises:
        If the input has rank < 2.
    """
    g = input.graph()
    if input.tensor_type().rank() < 2:
        raise "Can't get tril of Tensor with rank < 2"
    input_shape = ops.shape_of(input)
    N = input_shape[-2]  # Number of rows
    M = input_shape[-1]  # number of columns
    mask = tri(N, M, g.scalar(Int64(k)))
    return input.rebind("rows", "cols") * mask


def tri(rows: Symbol, cols: Symbol, k: Symbol) -> Symbol:
    """Returns a triangular mask matrix.

    Args:
        rows: Number of rows in the returned matrix.
        cols: Number of columns in the returned matrix.
        k: The diagonal at which values at and below are True, and values
          above are False.

    Returns:
        A Dtype.bool matrix.
    """
    g = rows.graph()

    int_dtype = rows.tensor_type().dtype
    step = g.scalar(1, int_dtype)

    row = (
        ops.range_fill(start=g.scalar(0, int_dtype), limit=rows, step=step)
        .rebind("rows")
        .reshape(-1, 1)
    )
    col = (
        ops.range_fill(start=-k, limit=(cols - k), step=step)
        .rebind("cols")
        .reshape(1, -1)
    )
    return ops.greater_equal(row, col)


@value
struct GroupedQueryAttention[dtype: DType]:
    """An attention layer that uses an intermediate number of key-value heads.

    Parameters:
        dtype: The DType of the weights and inputs to this layer.
    """

    var hyperparams: HyperParams
    var wkqv: Linear
    var out_proj: Linear

    def __call__(
        self,
        input: Symbol,
        attn_bias: Optional[Symbol] = None,
        k_cache: Optional[Symbol] = None,
        v_cache: Optional[Symbol] = None,
        is_causal: Bool = True,
    ) -> (Symbol, Symbol, Symbol):
        """Builds the GQA layer.

        Args:
            input: Encoded inputs.
            attn_bias: An additive bias to apply to the attention weights.
            k_cache: Cached computed keys for previous tokens.
            v_cache: Cached computed values for previous tokens. If
              `k_cache` is defined, `v_cache` must be defined as well.
            is_causal: Whether to apply a mask to the attention layer to ensure
                that the output tokens are only based on past positions.
        Returns:
            Attention outputs, new k_cache, and new v_cache.
        Raises:
            Error when `v_cache` is not defined when `k_cache` is defined.
        """
        g = input.graph()
        n_heads = self.hyperparams.n_heads
        kv_n_heads = self.hyperparams.kv_n_heads
        d_model = self.hyperparams.d_model
        batch_size = self.hyperparams.batch_size
        head_dim = d_model // n_heads

        qkv = self.wkqv(input)
        split = ops.split[3](
            qkv,
            sizes=(d_model, kv_n_heads * head_dim, kv_n_heads * head_dim),
            axis=2,
        )
        seq_len = Dim("seq_len")
        query = split[0].rebind(batch_size, seq_len, d_model)
        key = split[1].rebind(batch_size, seq_len, kv_n_heads * head_dim)
        value = split[2].rebind(batch_size, seq_len, kv_n_heads * head_dim)

        # Apply scaled dot product attention on the query, key and value.
        q = query.reshape(batch_size, seq_len, n_heads, head_dim)
        q = ops.transpose(q, 1, 2)

        k = key.reshape(batch_size, seq_len, kv_n_heads, head_dim)
        k = ops.transpose(k, 1, 2)
        k = ops.transpose(k, 2, 3)

        v = value.reshape(batch_size, seq_len, kv_n_heads, head_dim)
        v = ops.transpose(v, 1, 2)

        full_seq_len = Dim("full_seq_len")
        if k_cache:
            k_cache_value = k_cache.value()
            k = ops.concat(
                List[Symbol](k_cache_value, k), axis=3, out_dim=full_seq_len
            )
            if not v_cache:
                raise error(g, "v_cache cannot be None if k_cache is defined.")
            v_cache_value = v_cache.value()
            v = ops.concat(
                List[Symbol](v_cache_value, v), axis=2, out_dim=full_seq_len
            )

        k = k.rebind(batch_size, kv_n_heads, head_dim, full_seq_len)
        v = v.rebind(batch_size, kv_n_heads, full_seq_len, head_dim)

        # Record the k and v into the cache. An extra dimension is added
        # so that all cached keys/values can be concatenated on that dimension.
        k_cache_update = k.reshape(
            1, batch_size, kv_n_heads, head_dim, full_seq_len
        )
        v_cache_update = v.reshape(
            1, batch_size, kv_n_heads, full_seq_len, head_dim
        )

        if kv_n_heads > 1 and kv_n_heads < n_heads:
            # Repeat interleave k and v to match the number of heads in the
            # query.
            n_repeats = n_heads // kv_n_heads

            k = k.reshape(batch_size, kv_n_heads, 1, head_dim, full_seq_len)
            k = ops.tile(k, List[Int64](1, 1, n_repeats, 1, 1))
            k = k.reshape(
                batch_size, kv_n_heads * n_repeats, head_dim, full_seq_len
            )

            v = v.reshape(batch_size, kv_n_heads, 1, full_seq_len, head_dim)
            v = ops.tile(v, List[Int64](batch_size, 1, n_repeats, 1, 1))
            v = v.reshape(
                batch_size, kv_n_heads * n_repeats, full_seq_len, head_dim
            )

        softmax_scale = 1 / math.sqrt(d_model / n_heads)
        attn_weight = (q @ k) * softmax_scale
        s_q = ops.shape_of(q)[2]
        s_k = ops.shape_of(k)[-1]

        if attn_bias:
            bias = attn_bias.value()
            out_dims = List[Dim](1, n_heads, 1, full_seq_len)
            bias = bias[:, :, :, -s_k:, out_dims=out_dims]
            attn_weight = attn_weight + bias

        if is_causal and (not q.tensor_type().dims[2] == 1):
            # Apply a triangular mask to the attention weight so that in the
            # later matmul, each token in the ouput doesn't involve
            # information from future positions.
            s = ops.max(s_q, s_k)
            causal_mask = g.full[DType.bool](1, List[Symbol](s, s))
            causal_mask = tril(causal_mask)
            causal_mask = causal_mask[
                -s_q:, -s_k:, out_dims = List[Dim](seq_len, full_seq_len)
            ].reshape(1, 1, s_q, s_k)

            attn_weight_shape = ops.shape_of(attn_weight)

            min_val = g.op(
                "mo.broadcast_to",
                List[Symbol](g.scalar(min_finite[dtype]()), attn_weight_shape),
                TensorType(
                    dtype,
                    batch_size,
                    n_heads,
                    seq_len,
                    full_seq_len,
                ),
            )
            causal_mask = g.op(
                "mo.broadcast_to",
                List[Symbol](causal_mask, attn_weight_shape),
                TensorType(
                    causal_mask.tensor_type().dtype,
                    batch_size,
                    n_heads,
                    seq_len,
                    full_seq_len,
                ),
            )
            attn_weight = ops.select(causal_mask, attn_weight, min_val)

        attn_weight = ops.softmax(attn_weight)
        out = attn_weight @ v
        out = ops.transpose(out, 1, 2)
        out = out.reshape(batch_size, seq_len, d_model)
        return self.out_proj(out), k_cache_update, v_cache_update
