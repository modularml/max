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

import logging
from collections.abc import Sequence

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType, TensorValue
from max.pipelines import (
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextAndVisionContext,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)
from nn import Linear

from .conditional_generator import ConditionalGenerator
from .language_model import instantiate_language_model
from .vision_model import instantiate_vision_model


def max_seq_len(config: PipelineConfig) -> int:
    return min(
        config.max_length,
        config.huggingface_config.text_config.max_position_embeddings,
    )


class LlamaVision(PipelineModel):
    """The Llama3.2 vision model."""

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        cross_input_row_offsets: TensorValue,
        *kv_cache_inputs: TensorValue,
    ) -> TensorValue:
        """Builds the graph: all op staging happens here in __call__."""
        vision_config = self.pipeline_config.huggingface_config.vision_config
        text_config = self.pipeline_config.huggingface_config.text_config
        conditional_generator = ConditionalGenerator(
            pipeline_config=self.pipeline_config,
            vision_model=instantiate_vision_model(
                dtype=self.pipeline_config.dtype,
                image_size=vision_config.image_size,
                patch_size=vision_config.patch_size,
                supported_aspect_ratios=vision_config.supported_aspect_ratios,
                hidden_size=vision_config.hidden_size,
                max_num_tiles=vision_config.max_num_tiles,
                num_channels=vision_config.num_channels,
                norm_eps=vision_config.norm_eps,
                attention_heads=vision_config.attention_heads,
                num_hidden_layers=vision_config.num_hidden_layers,
                intermediate_size=vision_config.intermediate_size,
                num_global_layers=vision_config.num_global_layers,
                intermediate_layers_indices=vision_config.intermediate_layers_indices,
                weights=self.weights,
            ),
            multi_modal_projector=Linear(
                self.weights.multi_modal_projector.weight.allocate(
                    self.pipeline_config.dtype,
                    [
                        self.pipeline_config.huggingface_config.text_config.hidden_size,
                        self.pipeline_config.huggingface_config.vision_config.vision_output_dim,
                    ],
                ),
                self.weights.multi_modal_projector.bias.allocate(
                    self.pipeline_config.dtype,
                    [
                        self.pipeline_config.huggingface_config.text_config.hidden_size
                    ],
                ),
            ),
            language_model=instantiate_language_model(
                dtype=self.pipeline_config.dtype,
                hidden_size=text_config.hidden_size,
                n_heads=text_config.num_attention_heads,
                rope_theta=text_config.rope_theta,
                max_seq_len=max_seq_len(self.pipeline_config),
                num_hidden_layers=text_config.num_hidden_layers,
                cross_attention_layers=text_config.cross_attention_layers,
                vocab_size=text_config.vocab_size,
                rms_norm_eps=text_config.rms_norm_eps,
                num_key_value_heads=text_config.num_key_value_heads,
                intermediate_size=text_config.intermediate_size,
                kv_params=self._get_kv_params(),
                weights=self.weights,
            ),
        )

        return conditional_generator(
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            input_ids,
            hidden_input_row_offsets,
            cross_input_row_offsets,
            kv_cache_inputs,
        )

    def _llama3_vision_graph(self) -> Graph:
        # TODO: Verify if the mapping is correct:
        # From dumping the inputs before executing the reference model...
        # key: input_ids, shape: torch.Size([1, 14])
        # key: pixel_values, shape: torch.Size([1, 1, 4, 3, 448, 448])
        # key: aspect_ratio_ids, shape: torch.Size([1, 1])
        # key: aspect_ratio_mask, shape: torch.Size([1, 1, 4])
        # key: cross_attention_mask, shape: torch.Size([1, 14, 1, 4])

        # Inserted a manual CHW -> HWC transpose here.
        pixel_values_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                "batch_size",
                1,  # num_concurrent_media
                4,  # num_tiles
                3,  # num_channels
                448,  # height
                448,  # width
            ],
        )
        aspect_ratio_ids_type = TensorType(
            DType.int64,
            shape=[
                "batch_size",
                1,
            ],  # batch_size, num_concurrent_media
        )
        aspect_ratio_mask_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                "batch_size",
                1,
                4,
            ],  # batch_size, num_concurrent_media, num_tiles
        )

        input_ids_type = TensorType(DType.int64, shape=["total_seq_len"])
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"]
        )

        return Graph(
            "llama3-vision",
            # Pass PipelineModel as the Graph's forward callable until nn.Model.
            forward=self,
            input_types=[
                pixel_values_type,
                aspect_ratio_ids_type,
                aspect_ratio_mask_type,
                input_ids_type,
                # Pass 2 sets of offsets for hidden and cross attn states.
                input_row_offsets_type,
                input_row_offsets_type,
                *self.kv_manager.input_symbols()[0],
            ],
        )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        if self.pipeline_config.cache_strategy != KVCacheStrategy.CONTINUOUS:
            msg = "Llama Vision only supports continuous batching"
            raise ValueError(msg)

        batch_size = len(context_batch)
        pixel_values = Tensor.zeros(
            shape=[batch_size, 1, 4, 448, 448, 3],
            dtype=self.pipeline_config.dtype,
        )
        aspect_ratio_ids = Tensor.zeros(
            shape=[batch_size, 1], dtype=self.pipeline_config.dtype
        )
        aspect_ratio_mask = Tensor.zeros(
            shape=[batch_size, 1, 4], dtype=self.pipeline_config.dtype
        )

        # Input row offset type: ["input_row_offsets_len"], UInt32
        hidden_input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_ids = Tensor.from_numpy(tokens).to(self.pipeline_config.device)

        return (
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            input_ids,
            hidden_input_row_offsets,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError("not yet implemented.")

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        model_outputs = self.model.execute(
            *model_inputs, copy_inputs_to_device=False
        )
        assert not self.pipeline_config.enable_echo
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.text_config.num_attention_heads,
            head_dim=self.pipeline_config.huggingface_config.text_config.hidden_size
            // self.pipeline_config.huggingface_config.text_config.num_attention_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(self, session: InferenceSession) -> KVCacheManager:
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=max_seq_len(self.pipeline_config),
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
            devices=[self.pipeline_config.device],
            session=session,
        )

    def estimate_kv_cache_size(self) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=max_seq_len(self.pipeline_config),
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
            devices=[self.pipeline_config.device],
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        self.weights = self.pipeline_config.load_weights()

        logging.info("Building model...")
        graph = self._llama3_vision_graph()
        logging.info("Compiling...")
        model = session.load(
            graph,
            weights_registry=self.weights.allocated_weights,
        )
        return model
