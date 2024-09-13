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
from max.graph import ValueLike, ops, DimLike, TensorValue
from .mlp import Linear
from .rotary_embedding import OptimizedRotaryEmbedding
from ..kv_cache_params import KVCacheParams, KVCacheLayout
from ..kernels import (
    fused_qk_rope,
    fused_qkv_matmul,
    flash_attention,
    kv_cache_length,
)


def generate_attention_mask(
    attention_mask: ValueLike,
    start_pos: DimLike,
    seq_len: DimLike,
    activation_dtype: DType,
) -> TensorValue:
    """Computes Attention mask."""
    mask_val = ops.broadcast_to(
        ops.constant(float("-inf"), activation_dtype),
        shape=[seq_len, seq_len],
    )
    mask = ops.band_part(mask_val, -1, 0, exclude=True)

    zeros = ops.broadcast_to(
        ops.constant(0, activation_dtype),
        shape=[seq_len, start_pos],
    )

    x = ops.concat([zeros, mask], axis=1, new_dim="post_seq_len")

    select_mask = ops.cast(
        ops.broadcast_to(attention_mask, shape=x.shape), DType.bool
    )

    y = ops.broadcast_to(
        ops.constant(float("-inf"), activation_dtype), shape=x.shape
    )

    return ops.select(select_mask, x, y)


@dataclass
class OptimizedAttention:
    n_heads: int
    kv_params: KVCacheParams

    wqkv: ValueLike
    wo: Linear

    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        x: ValueLike,
        mask: ValueLike,
        k_cache: ValueLike,
        v_cache: ValueLike,
    ) -> ...:
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

        # Apply rope
        xq = ops.reshape(
            xq,
            [
                batch_size,
                seq_len,
                self.n_heads,
                self.kv_params.head_dim,
            ],
        )
        xq = fused_qk_rope(self.kv_params, xq, k_cache, self.rope.freqs_cis)

        if self.kv_params.layout == KVCacheLayout.BHSD:
            xq = ops.transpose(xq, 1, 2)

        # Calculate Flash Attention
        # TODO: MSDK-922 - Update the below to use start_pos effectively

        start_pos = kv_cache_length(self.kv_params, k_cache)
        attn_mask = generate_attention_mask(
            mask, start_pos, seq_len, self.kv_params.dtype
        )

        attn_out = flash_attention(
            self.kv_params,
            input=xq,
            k_cache=k_cache,
            v_cache=v_cache,
            attn_mask=attn_mask,
        )

        if self.kv_params.layout == KVCacheLayout.BHSD:
            attn_out = ops.transpose(attn_out, 1, 2)

        attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

        return self.wo(attn_out), k_cache, v_cache
