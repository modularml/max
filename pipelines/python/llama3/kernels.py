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

from .kv_cache_params import KVCacheParams
from .kv_cache import ContiguousKVCacheType
from max.graph import ops, ValueLike, TensorType, TensorValue
from max.dtype import DType


def fused_qkv_matmul(
    kv_params: KVCacheParams,
    input: ValueLike,
    wqkv: ValueLike,
    k_cache: ValueLike,
    v_cache: ValueLike,
) -> TensorValue:
    """Computes fused query, key and value projections."""

    op_name = f"fused_qkv_matmul_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_{kv_params.layout}"
    return ops.custom(
        op_name,
        [input, wqkv, k_cache, v_cache],
        [TensorType(dtype=kv_params.dtype, shape=input.shape).to_mlir()],
    )[0]


def fused_qk_rope(
    kv_params: KVCacheParams,
    input: ValueLike,
    k_cache: ValueLike,
    freqs_cis_2d: ValueLike,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings."""

    op_name = f"fused_qk_rope_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_{kv_params.layout}"
    return ops.custom(
        op_name,
        [input, k_cache, freqs_cis_2d],
        [TensorType(dtype=kv_params.dtype, shape=input.shape).to_mlir()],
    )[0]


def flash_attention(
    kv_params: KVCacheParams,
    input: ValueLike,
    k_cache: ValueLike,
    v_cache: ValueLike,
    attn_mask: ValueLike,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache"""

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_{kv_params.layout}"
    return ops.custom(
        op_name,
        [
            input,
            k_cache,
            v_cache,
            attn_mask,
            ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.int32)),
        ],
        [TensorType(dtype=kv_params.dtype, shape=input.shape).to_mlir()],
    )[0]


def kv_cache_length(
    kv_params: KVCacheParams, kv_cache: ValueLike
) -> TensorValue:
    """Calculate the length of the passed kv_cache collection."""

    op_name = f"kv_cache_length_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_{kv_params.layout}_{kv_params.dtype_shorthand}"
    return ops.custom(
        op_name,
        [kv_cache],
        [TensorType(dtype=DType.int8, shape=[]).to_mlir()],
    )[0]


def key_cache_for_layer(
    kv_params: KVCacheParams, i: int, kv_collection: ValueLike
) -> ContiguousKVCacheType:
    """Return the key cache for a specific layer from a collection."""

    op_name = f"key_cache_for_layer_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_{kv_params.layout}_{kv_params.dtype_shorthand}"
    return ops.custom(
        op_name,
        [ops.constant(i, dtype=DType.int8), kv_collection],
        [ContiguousKVCacheType.to_mlir()],
    )[0]


def value_cache_for_layer(
    kv_params: KVCacheParams, i: int, kv_collection: ValueLike
) -> ContiguousKVCacheType:
    """Return the value cache for a specific layer from a collection."""

    op_name = f"value_cache_for_layer_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_{kv_params.layout}_{kv_params.dtype_shorthand}"
    return ops.custom(
        op_name,
        [ops.constant(i, dtype=DType.int8), kv_collection],
        [ContiguousKVCacheType.to_mlir()],
    )[0]
