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

"""Llama 3.2 Transformer Vision Language Model."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.graph.weights import SafetensorWeights
from nn import Embedding, Linear, LPLayerNorm
from nn.layer import Layer

from .hyperparameters import TextHyperparameters


def lp_layer_norm(
    dtype: DType,
    size: int,
    eps: float,
    weights: SafetensorWeights,
) -> LPLayerNorm:
    """
    Helper function to instantiate a LPLayerNorm layer.
    """
    return LPLayerNorm(weights.weight.allocate(dtype, [size]), eps=eps)


# TODO: Copy pasted from other pipelines - maybe worth moving to a util subdir?
def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    """
    Helper function to instantiate a Linear layer.
    """
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


@dataclass
class TextModel(Layer):
    """
    The Llama text model which consists of transformer with self and cross attention layers.
    """

    params: TextHyperparameters
    embed_tokens: Embedding
    # TODO: This is essentially a nn.ModuleList
    # layers: list[CrossAttentionDecoderLayer | SelfAttentionDecoderLayer]
    # norm: TextRMSNorm
    # rotary_emb: RotaryEmbedding

    # input_ids: shape=[1, 1], dtype=torch.int64
    # attention_mask: shape=[1, 22], dtype=torch.int64
    # position_ids: shape=[1, 1], dtype=torch.int64
    # cross_attention_states: value=None
    # cross_attention_mask: shape=[1, 1, 1, 4100], dtype=torch.bfloat16
    # full_text_row_masked_out_mask: shape=[1, 1, 1, 1], dtype=torch.bfloat16
    # past_key_values: value=DynamicCache()
    # inputs_embeds: value=None
    # use_cache: value=True
    # output_attentions: value=False
    # output_hidden_states: value=False
    # return_dict: value=True
    def __call__(
        self,
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        past_key_values: Cache | list[TensorValue] | None = None,
        inputs_embeds: TensorValue | None = None,
        use_cache: bool | None = None,  # True
        cache_position: TensorValue | None = None,
    ) -> tuple:
        # TODO: Finish implementation.
        hidden_states = None
        next_cache = None
        all_hidden_states = None
        all_self_attns = None

        return (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_self_attns,
        )


@dataclass
class Cache(Layer):
    """
    TODO: This is just a stub to get the following code to pass. Remove it!
    """

    pass


@dataclass
class CausalLanguageModel(Layer):
    """
    The Llama Vision Text Model with a language modeling head on top.
    """

    params: TextHyperparameters
    model: TextModel
    lm_head: Linear

    # input_ids: shape=[1, 14], dtype=torch.int64
    # attention_mask: shape=[1, 14], dtype=torch.int64
    # position_ids: shape=[1, 14], dtype=torch.int64
    # cross_attention_states: shape=[4, 1025, 4096], dtype=torch.bfloat16
    # cross_attention_mask: shape=[1, 1, 14, 4100], dtype=torch.bfloat16
    # full_text_row_masked_out_mask: shape=[1, 1, 14, 1], dtype=torch.bfloat16
    # past_key_values: value=DynamicCache()
    # inputs_embeds: value=None
    # labels: value=None
    # use_cache: value=True
    # output_attentions: value=False
    # output_hidden_states: value=False
    # return_dict: value=True
    # cache_position: shape=[14], dtype=torch.int64
    # num_logits_to_keep: value=1
    def __call__(
        self,
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        past_key_values: Cache | list[TensorValue] | None = None,
        inputs_embeds: TensorValue | None = None,
        use_cache: bool | None = None,
        cache_position: TensorValue | None = None,
        num_logits_to_keep: int = 0,
    ) -> tuple:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        last_hidden_state, past_key_values, hidden_states, attentions = outputs
        logits = self.lm_head(
            last_hidden_state[:, -num_logits_to_keep:, :]
        ).float()

        return (
            None,  # TODO: loss. Maybe not needed at all?
            logits,
            past_key_values,
            hidden_states,
            attentions,
        )


def instantiate_language_model(
    params: TextHyperparameters,
    weights: SafetensorWeights,
) -> CausalLanguageModel:
    text_model = TextModel(
        params=params,
        embed_tokens=Embedding(
            weights.language_model.model.embed_tokens.weight.allocate(
                DType.bfloat16,
                [
                    params.vocab_size + 8,
                    params.hidden_size,
                ],
            ),
        ),
    )

    return CausalLanguageModel(
        params=params,
        model=text_model,
        lm_head=Linear(
            weights.language_model.lm_head.weight.allocate(
                DType.bfloat16, [params.vocab_size, params.hidden_size], None
            )
        ),
    )
