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
from max.graph import TensorValue, ops
from max.graph.weights import SafetensorWeights
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)
from nn import (
    MLP,
    AttentionWithRopeQKV,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
    TransformerBlock,
)
from nn.layer import Layer

from .cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)


@dataclass
class TextModel(Layer):
    """
    The Llama text model which consists of transformer with self and cross attention layers.
    """

    dtype: DType
    kv_params: KVCacheParams
    embed_tokens: Embedding
    layers: list[CrossAttentionDecoderLayer | TransformerBlock]
    norm: RMSNorm
    cross_attention_layers: list[int]
    rotary_emb: OptimizedRotaryEmbedding

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
        position_ids: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: TensorValue | None = None,
        cache_position: TensorValue | None = None,
        input_row_offset: TensorValue | None = None,
        **kwargs,
    ) -> tuple:
        inputs_embeds = self.embed_tokens(input_ids)

        # TODO: This should be removed. When we fix the hard-coded Dtypes.
        hidden_states = ops.cast(inputs_embeds, self.dtype)

        # causal_mask = self._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        #     output_attentions=False,
        # )
        # TODO: Finish implementation - stubbing out a bunch of outputs for now.
        # This causal_mask is only used by self attention, not cross attention.
        causal_mask = ops.constant(0, self.dtype).broadcast_to(
            (
                1,
                1,
                14,
                4100,
            )  # causal_mask / attention_mask: shape=[1, 1, 14, 4100]
        )

        # create position embeddings to be shared across the decoder layers
        # Inputs:
        # x: shape=[1, 1, 4096], dtype=torch.bfloat16
        # position_ids: shape=[1, 1], dtype=torch.int64
        # Output Shapes: [[1, 1, 128], [1, 1, 128]], torch.bfloat16
        # TODO(MAXCORE-119): Finish implementation
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = ops.constant(0, DType.bfloat16).broadcast_to(
            (
                1,
                1,
                128,
            )
        )
        position_embeddings = ops.stack(
            [position_embeddings, position_embeddings], axis=-1
        )

        # decoder layers
        all_hidden_states = None
        all_self_attns = None

        for idx, decoder_layer in enumerate(self.layers):
            # TODO: Implement this.
            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have
            # cross attention states or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
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

            # TODO: We need to check if the kwargs map 1:1 with the two different
            # *Attention layers here. Some are used in cross_attention, others in
            # self attention, most of them unused though
            hidden_states, kv_collection = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                cross_attention_mask=cross_attention_mask,
                attention_mask=causal_mask,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                kv_collection=kv_collection,
                valid_lengths=cache_lengths,
                input_row_offset=input_row_offset,
            )

        hidden_states = self.norm(hidden_states)

        return (
            hidden_states,
            all_hidden_states,
            all_self_attns,
        )


@dataclass
class CausalLanguageModel(Layer):
    """
    The Llama Vision Text Model with a language modeling head on top.
    """

    dtype: DType
    kv_params: KVCacheParams
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
        input_row_offset: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        cache_position: TensorValue | None = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> tuple:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        last_hidden_state, hidden_states, attentions = self.model(
            kv_cache_inputs=kv_cache_inputs,
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            cache_position=cache_position,
            input_row_offset=input_row_offset,
            **kwargs,
        )

        last_hidden_state, past_key_values, hidden_states, attentions = outputs

        # # For ragged tensors gather the last tokens from packed dim 0.
        # input_row_offset = kwargs["input_row_offset"]
        # last_token_indices = input_row_offset[1:] - 1  # type: ignore
        # # Should be: last_token = h[last_token_indices]
        # last_token = ops.gather(h, last_token_indices, axis=0)

        last_token_indices = input_row_offset[1:] - 1
        last_token_logits = ops.gather(
            last_hidden_state, last_token_indices, axis=0
        )
        logits = ops.cast(self.lm_head(last_token_logits), self.dtype)

        return (
            None,  # TODO: loss. Maybe not needed at all?
            logits,
            hidden_states,
            attentions,
        )


def cross_attention_decoder_layer(
    dtype: DType,
    num_attention_heads: int,
    hidden_size: int,
    num_key_value_heads: int,
    rms_norm_eps: float,
    kv_params: KVCacheParams,
    intermediate_size: int,
    weights: SafetensorWeights,
    layer_idx: int,
) -> CrossAttentionDecoderLayer:
    head_dim = hidden_size // num_attention_heads
    sdpa_attn = CrossSdpaAttention(
        n_heads=num_attention_heads,
        kv_params=kv_params,
        layer_idx=layer_idx,
        wq=weights.cross_attn.q_proj.weight.allocate(
            dtype,
            [
                num_attention_heads * head_dim,
                hidden_size,
            ],
        ),
        wk=weights.cross_attn.k_proj.weight.allocate(
            dtype,
            [
                num_key_value_heads * head_dim,
                hidden_size,
            ],
        ),
        wv=weights.cross_attn.v_proj.weight.allocate(
            dtype,
            [
                num_key_value_heads * head_dim,
                hidden_size,
            ],
        ),
        wo=Linear(
            weight=weights.cross_attn.o_proj.weight.allocate(
                dtype,
                [
                    hidden_size,
                    num_attention_heads * head_dim,
                ],
            ),
            bias=None,
        ),
        q_norm=RMSNorm(
            weight=weights.cross_attn.q_norm.weight.allocate(
                dtype,
                [head_dim],
            ),
            eps=rms_norm_eps,
        ),
        k_norm=RMSNorm(
            weight=weights.cross_attn.k_norm.weight.allocate(
                dtype,
                [head_dim],
            ),
            eps=rms_norm_eps,
        ),
    )
    return CrossAttentionDecoderLayer(
        cross_attn=sdpa_attn,
        input_layernorm=RMSNorm(
            weight=weights.input_layernorm.weight.allocate(
                dtype,
                [hidden_size],
            ),
            eps=rms_norm_eps,
        ),
        cross_attn_attn_gate=weights.cross_attn_attn_gate.allocate(
            dtype,
            [1],
        ),
        mlp=MLP(
            gate_proj=Linear(
                weight=weights.mlp.gate_proj.weight.allocate(
                    dtype,
                    [
                        intermediate_size,
                        hidden_size,
                    ],
                ),
                bias=None,
            ),
            down_proj=Linear(
                weight=weights.mlp.down_proj.weight.allocate(
                    dtype,
                    [
                        hidden_size,
                        intermediate_size,
                    ],
                ),
                bias=None,
            ),
            up_proj=Linear(
                weight=weights.mlp.up_proj.weight.allocate(
                    dtype,
                    [
                        intermediate_size,
                        hidden_size,
                    ],
                ),
                bias=None,
            ),
        ),
        post_attention_layernorm=RMSNorm(
            weight=weights.post_attention_layernorm.weight.allocate(
                dtype,
                [hidden_size],
            ),
            eps=rms_norm_eps,
        ),
        cross_attn_mlp_gate=weights.cross_attn_mlp_gate.allocate(
            dtype,
            [1],
        ),
    )


def self_attention_decoder_layer(
    dtype: DType,
    num_attention_heads: int,
    hidden_size: int,
    num_key_value_heads: int,
    intermediate_size: int,
    rms_norm_eps: float,
    kv_params: KVCacheParams,
    weights: SafetensorWeights,
    layer_idx: int,
    rotary_embedding: OptimizedRotaryEmbedding,
) -> TransformerBlock:
    head_dim = hidden_size / num_attention_heads

    q_proj = Linear(
        weight=weights.self_attn.q_proj.weight.allocate(
            dtype,
            [
                num_attention_heads * head_dim,
                hidden_size,
            ],
        ),
        bias=None,
    )
    k_proj = Linear(
        weight=weights.self_attn.k_proj.weight.allocate(
            dtype,
            [
                num_key_value_heads * head_dim,
                hidden_size,
            ],
        ),
        bias=None,
    )
    v_proj = Linear(
        weight=weights.self_attn.v_proj.weight.allocate(
            dtype,
            [
                num_key_value_heads * head_dim,
                hidden_size,
            ],
        ),
        bias=None,
    )
    o_proj = Linear(
        weight=weights.self_attn.o_proj.weight.allocate(
            dtype,
            [
                hidden_size,
                num_attention_heads * head_dim,
            ],
        ),
        bias=None,
    )

    attention = AttentionWithRopeQKV(
        n_heads=num_attention_heads,
        kv_params=kv_params,
        layer_idx=layer_idx,
        wq=q_proj.weight,
        wk=k_proj.weight,
        wv=v_proj.weight,
        wo=o_proj,
        rope=rotary_embedding,
    )
    return TransformerBlock(
        attention=attention,
        mlp=MLP(
            gate_proj=Linear(
                weight=weights.mlp.gate_proj.weight.allocate(
                    dtype,
                    [intermediate_size, hidden_size],
                ),
                bias=None,
            ),
            down_proj=Linear(
                weight=weights.mlp.down_proj.weight.allocate(
                    dtype,
                    [hidden_size, intermediate_size],
                ),
                bias=None,
            ),
            up_proj=Linear(
                weight=weights.mlp.up_proj.weight.allocate(
                    dtype,
                    [intermediate_size, hidden_size],
                ),
                bias=None,
            ),
        ),
        attention_norm=RMSNorm(
            weight=weights.input_layernorm.weight.allocate(
                dtype,
                [hidden_size],
            ),
            eps=rms_norm_eps,
        ),
        mlp_norm=RMSNorm(
            weight=weights.post_attention_layernorm.weight.allocate(
                dtype, [hidden_size]
            ),
            eps=rms_norm_eps,
        ),
    )


def instantiate_language_model(
    dtype: DType,
    hidden_size: int,
    n_heads: int,
    rope_theta: int,
    max_seq_len: int,
    num_hidden_layers: int,
    cross_attention_layers: list[int],
    vocab_size: int,
    rms_norm_eps: float,
    num_key_value_heads: int,
    intermediate_size: int,
    kv_params: KVCacheParams,
    weights: SafetensorWeights,
) -> CausalLanguageModel:
    layers: list[CrossAttentionDecoderLayer | TransformerBlock] = []

    # We don't really have a rotary embedding layer within the graph as it's largely
    # folded into the custom kernel, but leaving this here for now.
    rotary_embedding = OptimizedRotaryEmbedding(
        dim=hidden_size,
        n_heads=n_heads,
        theta=rope_theta,
        max_seq_len=max_seq_len,
        # TODO: Figure out how we want to pass this
        # rope_scaling=params.rope_scaling,
    )

    for layer_idx in range(
        num_hidden_layers,
    ):
        curr_layer_weight = weights.language_model.model.layers[layer_idx]

        if layer_idx in cross_attention_layers:
            layers.append(
                cross_attention_decoder_layer(
                    dtype=dtype,
                    num_attention_heads=n_heads,
                    hidden_size=hidden_size,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    kv_params=kv_params,
                    intermediate_size=intermediate_size,
                    weights=curr_layer_weight,
                    layer_idx=layer_idx,
                )
            )
        else:
            layers.append(
                self_attention_decoder_layer(
                    dtype=dtype,
                    num_attention_heads=n_heads,
                    hidden_size=hidden_size,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    kv_params=kv_params,
                    weights=curr_layer_weight,
                    layer_idx=layer_idx,
                    rotary_embedding=rotary_embedding,
                )
            )

    text_model = TextModel(
        dtype=dtype,
        kv_params=kv_params,
        embed_tokens=Embedding(
            weights.language_model.model.embed_tokens.weight.allocate(
                dtype,
                [
                    # Upstream in the Huggingface llama reference, 8 is added to the vocab size.
                    vocab_size + 8,
                    hidden_size,
                ],
            ),
        ),
        norm=RMSNorm(
            weight=weights.language_model.model.norm.weight.allocate(
                dtype,
                [hidden_size],
            ),
            eps=rms_norm_eps,
        ),
        layers=layers,
        cross_attention_layers=cross_attention_layers,
        # TODO: Verify if these values passed are even correct.
        rotary_emb=rotary_embedding,
    )

    return CausalLanguageModel(
        dtype=dtype,
        kv_params=kv_params,
        model=text_model,
        lm_head=Linear(
            weights.language_model.lm_head.weight.allocate(
                dtype,
                [
                    vocab_size,
                    hidden_size,
                ],
                None,
            )
        ),
    )
