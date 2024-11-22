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


import math
from dataclasses import dataclass
from typing import Tuple

from max.graph import TensorValue, TensorValueLike, ops
from nn.layer import Layer
from nn.linear import Linear

from .attention_utils import rotate_half


@dataclass
class Attention(Layer):
    n_heads: int
    dim: int  # hidden_size
    head_dim: int  # hidden_size // self.n_heads

    dropout: float

    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear

    def apply_rotary_embedding(
        self,
        xq: TensorValueLike,
        xk: TensorValueLike,
        cos: TensorValueLike,
        sin: TensorValueLike,
        unsqueeze_dim=0,
    ):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            xq (`TensorValueLike`): The query tensor.
            k (`TensorValueLike`): The key tensor.
            cos (`TensorValueLike`): The cosine part of the rotary embedding.
            sin (`TensorValueLike`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and
                sin so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos and sin have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos and sin broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(TensorValueLike)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = ops.unsqueeze(cos, unsqueeze_dim)
        sin = ops.unsqueeze(sin, unsqueeze_dim)
        xq = xq.transpose(1, 2)  # type: ignore
        xk = xk.transpose(1, 2)  # type: ignore

        q_embed = (xq * cos) + (rotate_half(xq) * sin)
        k_embed = (xk * cos) + (rotate_half(xk) * sin)
        return q_embed, k_embed

    def attention(
        self,
        xq: TensorValueLike,
        xk: TensorValueLike,
        xv: TensorValueLike,
        attn_mask: TensorValueLike,
    ) -> TensorValue:
        # Broadcast the attention mask across heads.
        # Do so in the graph so that the broadcast can be fused downstream ops.
        # batch, seq_len, post_seq_len = attn_mask.shape
        # attn_mask = attn_mask.reshape(
        #    (batch, 1, seq_len, post_seq_len)
        # ).broadcast_to((batch, self.n_heads, seq_len, post_seq_len))

        # xq = xq.transpose(0, 1)
        # xk = xk.transpose(0, 1)
        xv = xv.transpose(1, 2)  # type: ignore

        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        scores = ops.softmax(scores * scale + attn_mask)
        # scores = ops.dropout(scores, p=self.dropout)

        return scores @ xv

    def __call__(
        self,
        x: TensorValue,
        attention_mask: TensorValueLike,
        position_embeddings: Tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        """Computes attention on x, reusing the KV cache.

        Args:
            x: Activations with shape (batch, seq_len, dim).
            k_cache: The full keys cache buffer with shape
                (max_seq_len, n_layers, max_batch, n_heads, head_dim).
            v_cache: The full values cache buffer with shape
                (max_seq_len, n_layers, max_batch, n_heads, head_dim).
            start_pos: Scalar of the current position in the kv_cache.

        Returns the result of multi-headed self attention on the input.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # TODO: add transpose here or in apply_rotary_embedding?
        xq = ops.reshape(xq, [batch_size, seq_len, self.n_heads, self.head_dim])
        xk = ops.reshape(xk, [batch_size, seq_len, self.n_heads, self.head_dim])
        xv = ops.reshape(xv, [batch_size, seq_len, self.n_heads, self.head_dim])

        cos, sin = position_embeddings
        xq, xk = self.apply_rotary_embedding(xq, xk, cos, sin, unsqueeze_dim=0)

        output = (
            self.attention(xq, xk, xv, attention_mask)
            .transpose(1, 2)
            .reshape([batch_size, seq_len, -1])
        )
        return self.wo(output)
