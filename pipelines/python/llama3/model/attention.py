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
from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import DimLike, TensorValue, ValueLike, ops

from .mlp import Linear
from .rotary_embedding import RotaryEmbedding


def generate_attention_mask(
    attention_mask: ValueLike,
    start_pos: DimLike,
    seq_len: DimLike,
    activation_dtype: DType,
) -> TensorValue:
    """Computes Attention mask."""
    mask_val = ops.broadcast_to(
        ops.scalar(float("-inf"), activation_dtype),
        shape=[seq_len, seq_len],
    )
    mask = ops.band_part(mask_val, -1, 0, exclude=True)

    zeros = ops.broadcast_to(
        ops.scalar(0, activation_dtype),
        shape=[seq_len, start_pos],
    )

    x = ops.concat([zeros, mask], axis=1, new_dim="post_seq_len")

    select_mask = ops.cast(
        ops.broadcast_to(attention_mask, shape=x.shape), DType.bool
    )

    y = ops.broadcast_to(
        ops.scalar(float("-inf"), activation_dtype), shape=x.shape
    )

    return ops.select(select_mask, x, y)


@dataclass
class Attention:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dim: int

    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear

    rope: RotaryEmbedding

    def repeat_kv(self, kv: TensorValue):
        """Repeats key/value tensors to match the number of query heads."""
        batch = kv.shape[0]
        kv = ops.reshape(kv, [batch, -1, self.n_kv_heads, 1, self.head_dim])

        # TODO (MSDK-655): Should use a ops.tile() implementation here when available.
        num_tiles = [1, 1, 1, self.n_heads // self.n_kv_heads, 1]
        for i, n in enumerate(num_tiles):
            tiles = [kv for _ in range(n)]
            kv = ops.concat(tiles, axis=i)
        return ops.reshape(kv, [batch, -1, self.n_heads, self.head_dim])

    def attention(
        self,
        xq: ValueLike,
        xk: ValueLike,
        xv: ValueLike,
        attention_mask: ValueLike,
        k_cache: ValueLike,
        v_cache: ValueLike,
    ) -> TensorValue:
        prev_seq_len = k_cache.shape[0]
        batch, seq_len = xq.shape[:2]

        k_cache = ops.squeeze(k_cache, axis=1)
        v_cache = ops.squeeze(v_cache, axis=1)

        keys = ops.concat(
            [k_cache, xk.transpose(0, 1)], new_dim="post_seq_len"
        ).transpose(0, 1)
        values = ops.concat(
            [v_cache, xv.transpose(0, 1)], new_dim="post_seq_len"
        ).transpose(0, 1)

        keys = self.repeat_kv(keys)
        values = self.repeat_kv(values)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scale = math.sqrt(1.0 / self.head_dim)
        mask = generate_attention_mask(
            attention_mask, prev_seq_len, seq_len, xq.dtype
        )
        scores = xq @ ops.transpose(keys, 2, 3)
        return ops.softmax(scale * scores + mask) @ values

    def __call__(
        self,
        x: ValueLike,
        attention_mask: ValueLike,
        k_cache: ValueLike,
        v_cache: ValueLike,
    ) -> TensorValue:
        """Computes attention on x, reusing the KV cache.

        Args:
            x: Activations with shape (batch, seq_len, dim).
            k_cache: Previously computed keys with shape
                (prev_seq_len, 1, batch, n_kv_heads, head_dim).
            v_cache: Previously computed values with shape
                (prev_seq_len, 1, batch, n_kv_heads, head_dim).

        Returns the result of multi-headed self attention on the input.
        """
        batch, seq_len = x.shape[0], x.shape[1]
        start_pos = k_cache.shape[0]
        # matmul weights
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = ops.reshape(xq, [batch, seq_len, self.n_heads, self.head_dim])

        xk = ops.reshape(xk, [batch, seq_len, self.n_kv_heads, self.head_dim])
        xv = ops.reshape(xv, [batch, seq_len, self.n_kv_heads, self.head_dim])

        xq = self.rope(xq, start_pos, seq_len)
        xk = self.rope(xk, start_pos, seq_len)
        output = (
            self.attention(xq, xk, xv, attention_mask, k_cache, v_cache)
            .transpose(1, 2)
            .reshape([batch, seq_len, -1])
        )
        return self.wo(output), xk, xv
