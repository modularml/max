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
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)
from nn import MLP, Embedding, Linear, RMSNorm, TransformerBlock
from nn.layer import Layer

from .cache import Cache
from .cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)
from .hyperparameters import TextHyperparameters
from .rotary_embedding_2d import RotaryEmbedding2D
from .self_attention_decoder import SelfSdpaAttention


@dataclass
class TextModel(Layer):
    """
    The Llama text model which consists of transformer with self and cross attention layers.
    """

    kv_params: KVCacheParams
    params: TextHyperparameters
    embed_tokens: Embedding
    layers: list[CrossAttentionDecoderLayer | TransformerBlock]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding2D

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
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: TensorValue | None = None,
        cache_position: TensorValue | None = None,
    ) -> tuple:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the"
                " same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # if cache_position is None:
        #     past_seen_tokens = (
        #         past_key_values.get_seq_length() if past_key_values
        #         is not None else 0
        #     )
        #     cache_position = torch.arange(
        #         past_seen_tokens,
        #         past_seen_tokens + inputs_embeds.shape[1],
        #         device=inputs_embeds.device,
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        #     output_attentions=False,
        # )
        # TODO: Finish implementation - stubbing out a bunch of outputs for now.
        # This causal_mask is only used by self attention, not cross attention.
        causal_mask = ops.constant(0, DType.bfloat16).broadcast_to(
            (
                1,
                1,
                14,
                4100,
            )  # causal_mask / attention_mask: shape=[1, 1, 14, 4100]
        )

        # Create position embeddings to be shared across the decoder layers.
        position_embeddings = self.rotary_emb(
            x=hidden_states, position_ids=position_ids
        )

        # decoder layers
        all_hidden_states = None
        all_self_attns = None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            # TODO: Implement this.
            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have
            # cross attention states or cached cross attention states.
            is_cross_attention_layer = idx in self.params.cross_attention_layers
            is_cross_attention_cache_empty = past_key_values is None
            # or (
            #     past_key_values is not None
            #     and past_key_values.get_seq_length(idx) == 0
            # )

            if (
                is_cross_attention_layer
                and cross_attention_states is None
                and is_cross_attention_cache_empty
            ):
                continue

            kv_collection_constructor = (
                FetchContinuousBatchingKVCacheCollection(self.kv_params)
            )
            kv_collection = kv_collection_constructor(*kv_cache_inputs)

            _, cache_lengths, _, _ = kv_cache_inputs
            layer_outputs = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                cross_attention_mask=cross_attention_mask,
                attention_mask=causal_mask,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                kv_collection=kv_collection,
                valid_lengths=cache_lengths,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]

        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache

        return (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_self_attns,
        )


@dataclass
class CausalLanguageModel(Layer):
    """
    The Llama Vision Text Model with a language modeling head on top.
    """

    kv_params: KVCacheParams
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
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: TensorValue | None = None,
        cache_position: TensorValue | None = None,
        num_logits_to_keep: int = 0,
    ) -> tuple:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            kv_cache_inputs=kv_cache_inputs,
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        last_hidden_state, past_key_values, hidden_states, attentions = outputs
        logits = ops.cast(
            self.lm_head(last_hidden_state[:, -num_logits_to_keep:, :]),
            DType.bfloat16,
        )

        return (
            None,  # TODO: loss. Maybe not needed at all?
            logits,
            past_key_values,
            hidden_states,
            attentions,
        )


def cross_attention_decoder_layer(
    params: TextHyperparameters, weights: SafetensorWeights, layer_idx: int
) -> CrossAttentionDecoderLayer:
    num_heads = params.num_attention_heads
    head_dim = params.hidden_size // num_heads
    num_key_value_groups = num_heads // params.num_key_value_heads
    sdpa_attn = CrossSdpaAttention(
        params=params,
        num_heads=num_heads,
        num_key_value_heads=params.num_key_value_heads,
        head_dim=head_dim,
        layer_idx=layer_idx,
        num_key_value_groups=num_key_value_groups,
        q_proj=Linear(
            weight=weights.cross_attn.q_proj.weight.allocate(
                DType.bfloat16, [num_heads * head_dim, params.hidden_size]
            ),
            bias=None,
        ),
        k_proj=Linear(
            weight=weights.cross_attn.k_proj.weight.allocate(
                DType.bfloat16,
                [params.num_key_value_heads * head_dim, params.hidden_size],
            ),
            bias=None,
        ),
        v_proj=Linear(
            weight=weights.cross_attn.v_proj.weight.allocate(
                DType.bfloat16,
                [params.num_key_value_heads * head_dim, params.hidden_size],
            ),
            bias=None,
        ),
        o_proj=Linear(
            weight=weights.cross_attn.o_proj.weight.allocate(
                DType.bfloat16, [params.hidden_size, num_heads * head_dim]
            ),
            bias=None,
        ),
        q_norm=RMSNorm(
            weight=weights.cross_attn.q_norm.weight.allocate(
                DType.bfloat16,
                [head_dim],
            ),
            eps=params.rms_norm_eps,
        ),
        k_norm=RMSNorm(
            weight=weights.cross_attn.k_norm.weight.allocate(
                DType.bfloat16,
                [head_dim],
            ),
            eps=params.rms_norm_eps,
        ),
    )
    return CrossAttentionDecoderLayer(
        cross_attn=sdpa_attn,
        input_layernorm=RMSNorm(
            weight=weights.input_layernorm.weight.allocate(
                DType.bfloat16,
                [params.hidden_size],
            ),
            eps=params.rms_norm_eps,
        ),
        cross_attn_attn_gate=weights.cross_attn_attn_gate.allocate(
            DType.bfloat16,
            [1],
        ),
        mlp=MLP(
            gate_proj=Linear(
                weight=weights.mlp.gate_proj.weight.allocate(
                    DType.bfloat16,
                    [params.intermediate_size, params.hidden_size],
                ),
                bias=None,
            ),
            down_proj=Linear(
                weight=weights.mlp.down_proj.weight.allocate(
                    DType.bfloat16,
                    [params.hidden_size, params.intermediate_size],
                ),
                bias=None,
            ),
            up_proj=Linear(
                weight=weights.mlp.up_proj.weight.allocate(
                    DType.bfloat16,
                    [params.intermediate_size, params.hidden_size],
                ),
                bias=None,
            ),
        ),
        post_attention_layernorm=RMSNorm(
            weight=weights.post_attention_layernorm.weight.allocate(
                DType.bfloat16,
                [params.hidden_size],
            ),
            eps=params.rms_norm_eps,
        ),
        cross_attn_mlp_gate=weights.cross_attn_mlp_gate.allocate(
            DType.bfloat16,
            [1],
        ),
    )


def self_attention_decoder_layer(
    kv_params: KVCacheParams,
    params: TextHyperparameters,
    weights: SafetensorWeights,
    layer_idx: int,
) -> TransformerBlock:
    num_heads = params.num_attention_heads
    head_dim = params.hidden_size // num_heads

    q_proj = Linear(
        weight=weights.self_attn.q_proj.weight.allocate(
            DType.bfloat16, [num_heads * head_dim, params.hidden_size]
        ),
        bias=None,
    )
    k_proj = Linear(
        weight=weights.self_attn.k_proj.weight.allocate(
            DType.bfloat16,
            [params.num_key_value_heads * head_dim, params.hidden_size],
        ),
        bias=None,
    )
    v_proj = Linear(
        weight=weights.self_attn.v_proj.weight.allocate(
            DType.bfloat16,
            [params.num_key_value_heads * head_dim, params.hidden_size],
        ),
        bias=None,
    )

    attention = SelfSdpaAttention(
        n_heads=params.num_attention_heads,
        kv_params=kv_params,
        layer_idx=layer_idx,
        wq=q_proj.weight,
        wk=k_proj.weight,
        wv=v_proj.weight,
        wo=Linear(
            weight=weights.self_attn.o_proj.weight.allocate(
                DType.bfloat16, [params.hidden_size, num_heads * head_dim]
            ),
            bias=None,
        ),
    )
    return TransformerBlock(
        attention=attention,
        mlp=MLP(
            gate_proj=Linear(
                weight=weights.mlp.gate_proj.weight.allocate(
                    DType.bfloat16,
                    [params.intermediate_size, params.hidden_size],
                ),
                bias=None,
            ),
            down_proj=Linear(
                weight=weights.mlp.down_proj.weight.allocate(
                    DType.bfloat16,
                    [params.hidden_size, params.intermediate_size],
                ),
                bias=None,
            ),
            up_proj=Linear(
                weight=weights.mlp.up_proj.weight.allocate(
                    DType.bfloat16,
                    [params.intermediate_size, params.hidden_size],
                ),
                bias=None,
            ),
        ),
        attention_norm=RMSNorm(
            weight=weights.input_layernorm.weight.allocate(
                DType.bfloat16, [params.hidden_size]
            ),
            eps=params.rms_norm_eps,
        ),
        mlp_norm=RMSNorm(
            weight=weights.post_attention_layernorm.weight.allocate(
                DType.bfloat16, [params.hidden_size]
            ),
            eps=params.rms_norm_eps,
        ),
    )


def instantiate_language_model(
    kv_params: KVCacheParams,
    params: TextHyperparameters,
    weights: SafetensorWeights,
) -> CausalLanguageModel:
    layers: list[CrossAttentionDecoderLayer | TransformerBlock] = []

    for layer_idx in range(params.num_hidden_layers):
        curr_layer_weight = weights.language_model.model.layers[layer_idx]

        if layer_idx in params.cross_attention_layers:
            layers.append(
                cross_attention_decoder_layer(
                    params=params,
                    weights=curr_layer_weight,
                    layer_idx=layer_idx,
                )
            )
        else:
            layers.append(
                self_attention_decoder_layer(
                    kv_params=kv_params,
                    params=params,
                    weights=curr_layer_weight,
                    layer_idx=layer_idx,
                )
            )

    text_model = TextModel(
        kv_params=kv_params,
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
        layers=layers,
        norm=RMSNorm(
            weight=weights.language_model.model.norm.weight.allocate(
                DType.bfloat16,
                [params.hidden_size],
            ),
            eps=params.rms_norm_eps,
        ),
        # TODO: Verify if these values passed are even correct.
        rotary_emb=RotaryEmbedding2D(
            dim=params.hidden_size,
            n_heads=params.num_attention_heads,
            theta=params.rope_theta,
            max_patches_per_side=params.rope_scaling[
                "original_max_position_embeddings"
            ],
            # TODO: Figure out how we want to pass this
            # rope_scaling=params.rope_scaling,
        ),
    )

    return CausalLanguageModel(
        kv_params=kv_params,
        params=params,
        model=text_model,
        lm_head=Linear(
            weights.language_model.lm_head.weight.allocate(
                DType.bfloat16, [params.vocab_size, params.hidden_size], None
            )
        ),
    )
