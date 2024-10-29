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

"""Llama 3.2 Transformer Vision Encoder."""

from __future__ import annotations
from dataclasses import dataclass
from max.graph import TensorValue, TensorValueLike, ops
from nn import AttentionWithRope, LPLayerNorm
from nn.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    KVCacheParams,
)
from nn.layer import Layer
from .mlp import MLP


@dataclass
class VisionEncoderLayer(Layer):
    """
    This class implements a layer within Llama 3.2 vision transformer encoder.
    """

    # TODO: Integrate Attention
    # self_attn: AttentionWithRope
    mlp: MLP
    input_layernorm: LPLayerNorm
    post_attention_layernorm: LPLayerNorm
    is_gated: bool = False
    gate_attn: TensorValueLike | None = None
    gate_ffn: TensorValueLike | None = None

    def __call__(
        self,
        hidden_state: TensorValue,
        # TODO: Integrate kv_collection + attention args.
        # kv_collection: ContinuousBatchingKVCacheCollectionType,
        # valid_lengths: TensorValueLike,
        # **kwargs,
    ) -> tuple[TensorValue]:
        # Self Attention.
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)

        # TODO: Actually integrate attention layer. Making this a no-op for now.
        # hidden_state, kv_collection = self.self_attn(
        #     hidden_state,
        #     kv_collection,
        #     valid_lengths,
        #     **kwargs,
        # )

        if self.is_gated:
            hidden_state = ops.tanh(self.gate_attn) * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward.
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)

        # Gating criteria.
        if self.is_gated:
            hidden_state = ops.tanh(self.gate_ffn) * hidden_state
        hidden_state = residual + hidden_state
        outputs = (hidden_state,)

        return outputs


@dataclass
class VisionEncoder(Layer):
    """
    Transformer encoder consisting of # of hidden self attention
    layers. Each layer is a [`VisionEncoderLayer`].
    """

    # Called like so. is_gated is only used / propagated to the underlying VisionEncoderLayers.
    # self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
    # self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

    layers: list[VisionEncoderLayer]

    def __call__(
        self,
        hidden_states: TensorValueLike,
        # attention_mask: TensorValueLike | None = None,
        # output_hidden_states: bool | None = None,
    ):
        r"""
        Args:
            attention_mask (Tensor of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        for encoder_layer in self.layers:
            # TODO: Fix function call args.
            layer_outputs = encoder_layer(
                hidden_state=hidden_states,
                # attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        # Always return like that for now.
        return hidden_states

        # return_dict in reference implementation is always True, so we short
        # circuit this by removing the Union with tuple and just return BaseModelOutput.
        # TODO: Remove BaseModelOutput altogether.
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states,
        #     hidden_states=encoder_states,
        #     attentions=all_attentions,
        # )
