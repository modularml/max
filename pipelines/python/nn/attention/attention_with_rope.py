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

from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
)

from ..kernels import (
    flash_attention_ragged_with_causal_mask,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from ..rotary_embedding import OptimizedRotaryEmbedding
from .interfaces import AttentionImpl


@dataclass
class AttentionWithRope(AttentionImpl):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        x: TensorValueLike,
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        **kwargs,
    ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]:
        # Get attributes from input.
        total_seq_len = x.shape[0]  # type: ignore

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,  # type: ignore
            wqkv=self.wqkv,
            input_row_offset=kwargs["input_row_offset"],
            kv_collection=kv_collection,  # type: ignore
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
            kwargs["input_row_offset"],
            kv_collection,  # type: ignore
            freqs_cis,
            self.layer_idx,
        )

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged_with_causal_mask(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,  # type: ignore
            layer_idx=self.layer_idx,
            input_row_offset=kwargs["input_row_offset"],
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out), kv_collection  # type: ignore
