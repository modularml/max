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
from max.graph import TensorType, TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    KVCacheParams,
)


def fused_qkv_ragged_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offset: TensorValue,
    wqkv: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input.

    `input` and `input_row_offset` are used together to implement the ragged
    tensor.
    `input_row_offset` indicates where each batch starts and ends in `input`

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.dtype != wqkv.dtype:
        msg = (
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )
        raise ValueError(msg)

    if input.rank != 2:
        msg = f"expected input to have rank 2, was {input.rank}"
        raise ValueError(msg)

    if input_row_offset.dtype != DType.uint32:
        msg = (
            "expected input_row_offset to have dtype uint32, was"
            f" {input_row_offset.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    op_name = f"fused_qkv_matmul_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_cont_batch_ragged"

    return ops.custom(
        op_name,
        values=[input, input_row_offset, wqkv, kv_collection, layer_idx],  # type: ignore
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def fused_qkv_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    wqkv: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollectionType,
    layer_idx: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key and value projections."""

    if input.dtype != wqkv.dtype:
        msg = (
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )
        raise ValueError(msg)

    if input.rank != 3:
        msg = f"expected input to have rank 3, was {input.rank}"
        raise ValueError(msg)

    if wqkv.rank != 2:
        msg = f"expected wqkv to have rank 2, was {wqkv.rank}"
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    op_name = f"fused_qkv_matmul_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch"

    return ops.custom(
        op_name,
        values=[input, wqkv, kv_collection, layer_idx],  # type: ignore
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def fused_qk_ragged_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offset: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings and ragged inputs.

    `input` and `input_row_offset` are used together to implement the ragged tensor.
    `input_row_offset` indicates where each batch starts and ends in `input`
    """

    if input.dtype != freqs_cis.dtype:
        msg = (
            "expected input and freqs_cis to share a dtype, but got"
            f" {input.dtype} and {freqs_cis.dtype} respectively"
        )
        raise ValueError(msg)

    if input_row_offset.dtype != DType.uint32:
        msg = (
            "expected input_row_offset to have dtype uint32, was"
            f" {input_row_offset.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    op_name = f"fused_qk_rope_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch_ragged"

    return ops.custom(
        op_name,
        values=[input, input_row_offset, kv_collection, freqs_cis, layer_idx],  # type: ignore
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def fused_qk_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis_2d: TensorValue,
    layer_idx: TensorValue,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings."""
    op_name = f"fused_qk_rope_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch"

    return ops.custom(
        op_name,
        values=[input, kv_collection, freqs_cis_2d, layer_idx],  # type: ignore
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def flash_attention(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    attention_mask: TensorValue,
    valid_lengths: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache."""
    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_bshd_continuous_batch"

    # NOTE: The scale argument to the flash attention kernel is constrained to
    # float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))
    return ops.custom(
        op_name,
        values=[
            input,
            kv_collection,  # type: ignore
            layer_idx,
            attention_mask,
            valid_lengths,
            scale,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def flash_attention_with_causal_mask(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache.
    Notably, materializes the causal mask within the kernel."""

    if input.shape[0] != valid_lengths.shape[0]:
        msg = (
            "expected batch size of input, to equal length of valid_lengths"
            f" got batch size of input ({input.shape[0]}), length of"
            f" valid_lengths ({valid_lengths.shape[0]})"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if valid_lengths.dtype != DType.uint32:
        msg = f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        raise ValueError(msg)

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_causal_mask_continuous_batch"

    # NOTE: The scale argument to flash attention is constrained to float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))
    return ops.custom(
        op_name,
        values=[input, kv_collection, layer_idx, valid_lengths, scale],  # type: ignore
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def flash_attention_ragged_with_causal_mask(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offset: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache.
    Notably, materializes the causal mask within the kernel.

    `input` and `input_row_offset` are used together to implement the ragged tensor.
    `input_row_offset` indicates where each batch starts and ends in `input`
    """

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offset.dtype != DType.uint32:
        msg = (
            f"expected uint32 input_row_offset but got {input_row_offset.dtype}"
        )
        raise ValueError(msg)

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads}_d{kv_params.head_dim}_cont_batch_ragged"

    # NOTE: The scale argument to flash attention is constrained to float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))
    return ops.custom(
        op_name,
        values=[input, input_row_offset, kv_collection, layer_idx, scale],  # type: ignore
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0]


def swish_glu(
    a: TensorValueLike, b0: TensorValueLike, b1: TensorValueLike
) -> TensorValue:
    """Computes swish(a@b0.t()) * (a@b1.t())"""
    a = TensorValue(a)
    b0 = TensorValue(b0)
    b1 = TensorValue(b1)
    if a.rank != 2:
        msg = f"expected a to have rank 2, was {a.rank}"
        raise ValueError(msg)

    if b0.rank != 2:
        msg = f"expected b0 to have rank 2, was {b0.rank}"
        raise ValueError(msg)

    if b1.rank != 2:
        msg = f"expected b1 to have rank 2, was {b1.rank}"
        raise ValueError(msg)

    m = a.shape[0]
    n = b0.shape[0]
    if b0.shape[1] != a.shape[1]:
        msg = f"a.shape[1] == {a.shape[1]} != {b0.shape[1]} == b0.shape[1]"
        raise ValueError(msg)

    if b0.shape != b1.shape:
        msg = f"b0.shape == {b0.shape} != {b1.shape} == b1.shape"
        raise ValueError(msg)

    if a.dtype != b0.dtype or a.dtype != b1.dtype:
        msg = (
            "Element types of all arguments must be equal, but received"
            f" {a.dtype}, {b0.dtype}, and {b1.dtype}."
        )
        raise ValueError(msg)

    return ops.custom(
        "swishGLU",
        values=[a, b0, b1],
        out_types=[
            TensorType(
                dtype=a.dtype,
                shape=[m, n],
                device=a.device,
            )
        ],
    )[0].tensor
