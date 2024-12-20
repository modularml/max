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
"""An opaque KV Cache optimized attention mechanism with Rope."""

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, ops
from max.pipelines.kv_cache import ContinuousBatchingKVCacheCollection

from ..kernels import (
    flash_attention_ragged_with_causal_mask,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from ..rotary_embedding import OptimizedRotaryEmbedding
from .interfaces import AttentionImpl, AttentionImplQKV


@dataclass
class AttentionWithRope(AttentionImpl):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            n_heads=self.n_heads,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Cast freqs_cis to xq's dtype to match the fused_qk_ragged_rope kernel.
        freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            self.layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged_with_causal_mask(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)


@dataclass
class AttentionWithRopeQKV(AttentionImplQKV):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = ops.concat((self.wq, self.wk, self.wv), axis=0).transpose(0, 1)

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            n_heads=self.n_heads,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Cast freqs_cis to xq's dtype to match the fused_qk_ragged_rope kernel.
        freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            ops.constant(self.layer_idx, DType.uint32),
            interleaved=self.rope.interleaved,
        )

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged_with_causal_mask(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            input_row_offsets=kwargs["input_row_offsets"],
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)
