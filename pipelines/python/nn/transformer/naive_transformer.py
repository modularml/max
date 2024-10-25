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

from __future__ import annotations
from dataclasses import dataclass

from max.graph import TensorValueLike, TensorValue, ops

from ..attention import NaiveAttentionWithRope
from ..layer import Layer
from ..linear import MLP, Linear
from ..norm import RMSNorm
from ..embedding import Embedding


@dataclass
class NaiveTransformerBlock(Layer):
    """Max-Graph Only Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: NaiveAttentionWithRope
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: TensorValueLike,
        attention_mask: TensorValueLike,
        k_cache: TensorValueLike,
        v_cache: TensorValueLike,
        start_pos: TensorValue,
        layer_index: int,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        attention_out = self.attention(
            self.attention_norm(x),
            attention_mask,
            k_cache,
            v_cache,
            start_pos,
            layer_index,
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h


@dataclass
class NaiveTransformer(Layer):
    """Max-Graph only model consisting of NaiveTransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[NaiveTransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding

    def __call__(
        self,
        tokens: TensorValueLike,
        attention_mask: TensorValueLike,
        k_cache: TensorValueLike,
        v_cache: TensorValueLike,
        start_pos: TensorValueLike,
    ) -> tuple[TensorValue, TensorValue]:
        h = self.embedding(tokens)

        for i in range(len(self.layers)):
            h = self.layers[i](
                h,
                attention_mask,
                k_cache,
                v_cache,
                start_pos,
                i,
            )

        seq_len = TensorValue(tokens.shape[1])
        return (
            ops.cast(self.output(self.norm(h)), k_cache.dtype),
            start_pos + seq_len,
        )
