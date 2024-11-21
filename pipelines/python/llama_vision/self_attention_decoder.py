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

"""Llama 3.2 Transformer Vision Language Model self attention decoder."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
)
from nn import AttentionQKV
from nn.kernels import flash_attention, fused_qkv_matmul


@dataclass
class SelfSdpaAttention(AttentionQKV):
    """
    This is an overloaded Attention class that has a rotary embedding operation
    in between.
    """

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

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : int(x.shape[-1] // 2)]
            x2 = x[..., int(x.shape[-1] // 2) :]
            return ops.concat((-x2, x1), axis=-1)

        cos = ops.unsqueeze(cos, unsqueeze_dim)
        sin = ops.unsqueeze(sin, unsqueeze_dim)
        xq = xq.transpose(1, 2)  # type: ignore
        xk = xk.transpose(1, 2)  # type: ignore

        q_embed = (xq * cos) + (rotate_half(xq) * sin)
        k_embed = (xk * cos) + (rotate_half(xk) * sin)
        return q_embed, k_embed

    def __call__(
        self,
        x: TensorValue,  # type: ignore
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        **kwargs,
    ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]:
        if "attention_mask" not in kwargs:
            raise ValueError("attention_mask not passed as input to Attention")
        attention_mask = kwargs["attention_mask"]
        if attention_mask.dtype != x.dtype:
            msg = (
                "expected attention_mask and x to have the same dtype, but got"
                f" {attention_mask.dtype} and {x.dtype}, respectively."
            )
            raise ValueError(msg)

        # TODO: This positional embedding should happen after calling Linear...
        cos, sin = kwargs["position_embeddings"]
        # self.xq, self.xk = self.apply_rotary_embedding(
        #     self.xq, self.xk, cos, sin, unsqueeze_dim=0
        # )

        wqkv = ops.concat((self.wq, self.wk, self.wv), axis=0).transpose(0, 1)

        # Get attributes from inputs
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Call into fused qkv matmul.
        xq = fused_qkv_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            n_heads=self.n_heads,
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
            kv_collection=kv_collection,  # type: ignore
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            attention_mask=attention_mask,
            valid_lengths=kwargs["valid_lengths"],
        )

        attn_out = ops.reshape(attn_out, shape=[batch_size, seq_len, -1])

        return self.wo(attn_out), kv_collection  # type: ignore
