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
"""Helper functions for wrapping custom kv cache/attention related ops."""

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops

from .kv_cache import (
    ContinuousBatchingKVCache,
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheType,
    KVCacheParams,
)


def fused_qkv_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    wqkv: TensorValue,
    k_cache: ContinuousBatchingKVCache,
    v_cache: ContinuousBatchingKVCache,
) -> TensorValue:
    """Computes fused query, key and value projections."""
    op_name = f"fused_qkv_matmul_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch"

    return ops.custom(
        op_name,
        [input, wqkv, k_cache, v_cache],
        [TensorType(dtype=input.dtype, shape=input.shape)],
    )[0]


def fused_qk_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    k_cache: ContinuousBatchingKVCache,
    freqs_cis_2d: TensorValue,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings."""
    op_name = f"fused_qk_rope_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch"

    return ops.custom(
        op_name,
        [input, k_cache, freqs_cis_2d],
        [TensorType(dtype=input.dtype, shape=input.shape)],
    )[0]


def flash_attention(
    kv_params: KVCacheParams,
    input: TensorValue,
    k_cache: ContinuousBatchingKVCache,
    v_cache: ContinuousBatchingKVCache,
    attn_mask: TensorValue,
    valid_lengths: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache."""
    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch"

    # NOTE: The scale argument to the flash attentionkernel is constrained to float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))
    return ops.custom(
        op_name,
        [input, k_cache, v_cache, attn_mask, valid_lengths, scale],
        [TensorType(dtype=input.dtype, shape=input.shape)],
    )[0]


def key_cache_for_layer(
    kv_params: KVCacheParams,
    i: int,
    kv_collection: ContinuousBatchingKVCacheCollection,
) -> ContinuousBatchingKVCacheType:
    """Returns the key cache for a specific layer from a collection."""
    op_name = f"key_cache_for_layer_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_{kv_params.dtype_shorthand}_continuous_batch"

    return ops.custom(
        op_name,
        [ops.constant(i, dtype=DType.int64), kv_collection],
        [ContinuousBatchingKVCacheType()],
    )[0]


def value_cache_for_layer(
    kv_params: KVCacheParams,
    i: int,
    kv_collection: ContinuousBatchingKVCacheCollection,
) -> ContinuousBatchingKVCacheType:
    """Returns the value cache for a specific layer from a collection."""
    op_name = f"value_cache_for_layer_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_{kv_params.dtype_shorthand}_continuous_batch"

    return ops.custom(
        op_name,
        [ops.constant(i, dtype=DType.int64), kv_collection],
        [ContinuousBatchingKVCacheType()],
    )[0]
