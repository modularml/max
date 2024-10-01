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
"""An Opaque KV Cache Optimized attention mechanism."""

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, ValueLike, ops
from max.graph.type import Dim, DimLike

from .kernels import (
    ContiguousKVCache,
    flash_attention,
    fused_qk_rope,
    fused_qkv_matmul,
)

from .kv_caching import KVCacheLayout, KVCacheParams
from .layer import Layer
from .mlp import Linear
from .rotary_embedding import OptimizedRotaryEmbedding


def generate_attention_mask(
    attention_mask: TensorValue,
    start_pos: TensorValue,
    seq_len: DimLike,
    activation_dtype: DType,
) -> TensorValue:
    """Computes attention mask.

    Returns:
        A causal attention mask with -inf in the lower triangle.
    """
    fill_val = -10000.0
    batch, n_heads = attention_mask.shape[0], attention_mask.shape[1]
    mask_val = ops.broadcast_to(
        ops.constant(fill_val, activation_dtype),
        shape=[batch, n_heads, seq_len, seq_len],
    )

    # Create a lower triangular matrix filled with -inf.
    mask = ops.band_part(mask_val, num_lower=-1, num_upper=0, exclude=True)

    # Broadcast zero to (seq_len, start_pos).
    seq_len_val = ops.shape_to_tensor((seq_len,)).reshape(())
    zeros = ops.broadcast_to(
        ops.constant(0, activation_dtype),
        shape=ops.stack(
            [
                ops.shape_to_tensor((batch,)).reshape(()),
                ops.shape_to_tensor((n_heads,)).reshape(()),
                seq_len_val,
                ops.cast(start_pos, seq_len_val.dtype),
            ]
        ),
        out_dims=(batch, n_heads, "seq_len", "start_pos"),
    )

    x = ops.concat([zeros, mask], axis=-1, new_dim="post_seq_len")

    y = ops.broadcast_to(
        ops.constant(fill_val, activation_dtype), shape=x.shape
    )

    return ops.select(attention_mask, x, y)


def _dim_to_scalar(dim: Dim) -> TensorValue:
    return ops.shape_to_tensor((dim,)).reshape(())


@dataclass
class OptimizedAttention(Layer):
    n_heads: int
    kv_params: KVCacheParams

    wqkv: TensorValue
    wo: Linear

    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        x: TensorValue,
        mask: ValueLike,
        k_cache: ContiguousKVCache,
        v_cache: ContiguousKVCache,
        start_pos: TensorValue,
        valid_lengths: TensorValue,
    ) -> tuple[TensorValue, ContiguousKVCache, ContiguousKVCache]:
        # Get attributes from input.
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Call into fused qkv matmul.
        xq = fused_qkv_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            k_cache=k_cache,
            v_cache=v_cache,
        )

        # Apply rope.
        xq = ops.reshape(
            xq,
            [
                batch_size,
                seq_len,
                self.n_heads,
                self.kv_params.head_dim,
            ],
        )

        # Cast freqs_cis to xq's dtype to match the fused_qk_rope kernel.
        freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)

        # Slice out `freqs_cis` for our current position in the sequence.
        freqs_cis = freqs_cis[
            (
                slice(start_pos, start_pos + _dim_to_scalar(seq_len)),
                "seq_len",
            ),
            :,
        ]
        xq = fused_qk_rope(self.kv_params, xq, k_cache, freqs_cis)

        if self.kv_params.layout == KVCacheLayout.BHSD:
            xq = ops.transpose(xq, 1, 2)

        # Calculate Flash Attention.
        attn_mask = generate_attention_mask(
            mask, start_pos, seq_len, self.kv_params.dtype
        )

        attn_out = flash_attention(
            self.kv_params,
            input=xq,
            k_cache=k_cache,
            v_cache=v_cache,
            attn_mask=attn_mask,
            valid_lengths=valid_lengths,
        )

        if self.kv_params.layout == KVCacheLayout.BHSD:
            attn_out = ops.transpose(attn_out, 1, 2)

        attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

        return self.wo(attn_out), k_cache, v_cache
