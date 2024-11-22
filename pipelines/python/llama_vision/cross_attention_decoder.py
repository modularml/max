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

"""Llama 3.2 Transformer Vision Language Model cross attention decoder."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import ContinuousBatchingKVCacheCollectionType
from nn import Linear, MLP, RMSNorm
from nn.layer import Layer

from .cache import Cache


@dataclass
class CrossSdpaAttention(Layer):
    pipeline_config: PipelineConfig
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_idx: int
    num_key_value_groups: int

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear

    q_norm: RMSNorm
    k_norm: RMSNorm

    def __call__(
        self,
        hidden_states: TensorValue,
        cross_attention_states: TensorValue | None = None,
        past_key_value: Cache | None = None,
        attention_mask: TensorValue | None = None,
        use_cache: bool | None = None,
        cache_position: TensorValue | None = None,
    ) -> tuple[TensorValue, None, Cache | None]:
        # TODO: Stubbed out for now.
        attn_output = hidden_states

        return attn_output, None, past_key_value


@dataclass
class CrossAttentionDecoderLayer(Layer):
    """Cross-attention transformer block with tanh-gated attention and feedforward.
    """

    cross_attn: CrossSdpaAttention
    input_layernorm: RMSNorm
    cross_attn_attn_gate: TensorValueLike
    mlp: MLP
    post_attention_layernorm: RMSNorm
    cross_attn_mlp_gate: TensorValueLike

    def __call__(
        self,
        hidden_states: TensorValue,
        cross_attention_states: TensorValue,
        cross_attention_mask: TensorValue,
        attention_mask: TensorValue,  # unused in cross attention.
        # need to make this optional for now.
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        position_ids: TensorValue | None = None,  # unused in cross attention.
        past_key_value: Cache | None = None,
        cache_position: TensorValue | None = None,
        # unused in cross attention.
        position_embeddings: TensorValue | None = None,
        # unused in cross attention, for now.
        kv_collection: ContinuousBatchingKVCacheCollectionType | None = None,
        valid_lengths: int | None = None,  # unused in cross attention.
    ) -> tuple[TensorValue, Cache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = (
            residual + ops.tanh(self.cross_attn_attn_gate) * hidden_states
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = (
            residual + ops.tanh(self.cross_attn_mlp_gate) * hidden_states
        )

        return (hidden_states, past_key_value)
