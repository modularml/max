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
"""The Llama 3.2 model which consists of a vision encoder and a language model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import TensorValue
from nn import Linear
from nn.layer import Layer

from .hyperparameters import TextHyperparameters, VisionHyperparameters
from .language_model import CausalLanguageModel
from .vision_model import VisionModel


@dataclass
class ConditionalGenerator(Layer):
    """
    The Llama model which consists of a vision encoder and a language model.
    """

    text_params: TextHyperparameters
    vision_params: VisionHyperparameters
    vision_model: VisionModel
    multi_modal_projector: Linear
    language_model: CausalLanguageModel

    def prepare_cross_attention_mask(
        self,
        cross_attention_mask: np.ndarray | None = None,
        full_text_row_masked_out_mask: np.ndarray | None = None,
        cache_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = (
                self._prepare_cross_attention_mask_helper(
                    cross_attention_mask,
                    num_vision_tokens=self.vision_params.num_patches,
                    dtype=DType.bfloat16,
                )
            )
        else:
            full_text_row_masked_out_mask = None

        if (
            cross_attention_mask is not None
            and cache_position is not None
            and full_text_row_masked_out_mask is not None
        ):
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[
                :, :, cache_position
            ]
        return cross_attention_mask, full_text_row_masked_out_mask

    def _prepare_cross_attention_mask_helper(
        self,
        cross_attention_mask: np.ndarray,
        num_vision_tokens: int,
        dtype: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        # reshape so it can be used by attn module
        batch_size, text_total_length, *_ = cross_attention_mask.shape
        cross_attention_mask = np.repeat(
            cross_attention_mask, num_vision_tokens, axis=2
        )  # Repeat along the 3rd dimension
        cross_attention_mask = cross_attention_mask.reshape(
            batch_size, text_total_length, -1
        )
        cross_attention_mask = np.expand_dims(cross_attention_mask, axis=1)

        # invert the mask
        inverted_cross_attn_mask = (1.0 - cross_attention_mask).astype(dtype)
        cross_attention_mask = np.where(
            inverted_cross_attn_mask.astype(bool),
            np.finfo(dtype).min,
            inverted_cross_attn_mask,
        )

        # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
        # last dimension contains negative infinity values, otherwise it's 1
        negative_inf_value = np.finfo(dtype).min
        full_text_row_masked_out_mask = (
            (cross_attention_mask != negative_inf_value)
            .any(axis=-1, keepdims=True)
            .astype(dtype)
        )
        cross_attention_mask *= full_text_row_masked_out_mask

        return cross_attention_mask, full_text_row_masked_out_mask

    def __call__(
        self,
        pixel_values: TensorValue | None = None,
        aspect_ratio_ids: TensorValue | None = None,
        aspect_ratio_mask: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        cross_attention_states: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ]
        | None = None,
        input_ids: TensorValue | None = None,
        inputs_embeds: TensorValue | None = None,
        # TODO: Remove this since we have our own caching mechanism.
        cache_position: TensorValue | None = None,
        # For preparing cross attention mask.
        full_text_row_masked_out_mask: TensorValue | None = None,
    ) -> TensorValue:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the"
                " same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the"
                " same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError(
                "`pixel_values` and `cross_attention_states` cannot be provided"
                " simultaneously"
            )

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError(
                    "`aspect_ratio_ids` must be provided if `pixel_values` is"
                    " provided"
                )
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs[0]

            num_patches = cross_attention_states.shape[-2]
            cross_attention_states = self.multi_modal_projector(
                cross_attention_states
            ).reshape(
                (
                    -1,
                    num_patches,
                    self.text_params.hidden_size,
                )
            )

        # TODO: Remove this. I had to make it an optional so it respects the order
        # of arg inputs when unwrapping the graph inputs.
        assert kv_cache_inputs is not None, (
            "kv_cache_inputs is None. This should be impossible as it should"
            " already be instantiated during pipeline construction."
        )

        return self.language_model(
            kv_cache_inputs=kv_cache_inputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=None,  # DynamicCache()
            inputs_embeds=None,
            cache_position=cache_position,
            num_logits_to_keep=1,
        )
