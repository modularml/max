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
"""A vanilla opaque KV Cache optimized attention mechanism."""

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops

from ..layer import Layer
from ..mlp import Linear
from ..kernels import flash_attention, fused_qkv_matmul
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    KVCacheParams,
    KVCacheStrategy,
)


@dataclass
class Attention(Layer):
    n_heads: int
    kv_params: KVCacheParams
    layer_idx: TensorValue

    wqkv: TensorValue
    wo: Linear

    def __post_init__(self) -> None:
        if self.kv_params.cache_strategy == KVCacheStrategy.NAIVE:
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    def __call__(
        self,
        x: TensorValue,
        attn_mask: TensorValueLike,
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        valid_lengths: TensorValue,
    ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]:
        # Get attributes from inputs
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Call into fused qkv matmul.
        xq = fused_qkv_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
        )

        xq = ops.reshape(
            xq,
            [
                batch_size,
                seq_len,
                self.n_heads,
                self.kv_params.head_dim,
            ],
        )

        # Calculate Flash Attention
        attn_out = flash_attention(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            attn_mask=attn_mask,
            valid_lengths=valid_lengths,
        )

        attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

        return self.wo(attn_out), kv_collection
