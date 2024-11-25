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
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.pipelines import TextAndVisionContext, PipelineConfig, PipelineModel
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    load_kv_manager,
)
from nn import Linear

from .conditional_generator import ConditionalGenerator

from .language_model import instantiate_language_model
from .vision_model import instantiate_vision_model


# TODO: These are configured for text only model. What about vision model?
def load_llama_vision_and_kv_manager(
    config: PipelineConfig, session: InferenceSession | None = None
) -> tuple[LlamaVision, KVCacheManager]:
    # Initialize kv cache params and manager
    kv_params = KVCacheParams(
        dtype=config.dtype,
        n_kv_heads=config.huggingface_config.text_config.num_attention_heads,
        head_dim=config.huggingface_config.text_config.hidden_size
        // config.huggingface_config.text_config.num_attention_heads,
        cache_strategy=config.cache_strategy,
    )

    if session is None:
        session = InferenceSession(devices=[config.device])

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=config.max_cache_batch_size,  # verify this.
        max_seq_len=config.huggingface_config.text_config.max_position_embeddings,  # verify this.
        num_layers=config.huggingface_config.text_config.num_hidden_layers,
        devices=[config.device],
        session=session,
    )
    model = LlamaVision(
        pipeline_config=config,
        session=session,
    )

    return model, kv_manager


# TODO: Some parts may be consolidated under the parent Llama 3 pipeline interface.
class LlamaVision(PipelineModel):
    """The Llama3.2 vision model."""

    def export_mef(self, export_path):
        self._model._export_mef(export_path)

    def _llama3_vision_graph(
        self,
    ) -> Graph:
        # TODO: Verify if the mapping is correct:
        # From dumping the inputs before executing the reference model...
        # key: input_ids, shape: torch.Size([1, 14])
        # key: attention_mask, shape: torch.Size([1, 14])
        # key: pixel_values, shape: torch.Size([1, 1, 4, 3, 448, 448])
        # key: aspect_ratio_ids, shape: torch.Size([1, 1])
        # key: aspect_ratio_mask, shape: torch.Size([1, 1, 4])
        # key: cross_attention_mask, shape: torch.Size([1, 14, 1, 4])

        # Inserted a manual CHW -> HWC transpose here.
        pixel_values_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                1,  # batch_size
                1,  # num_concurrent_media
                4,  # num_tiles
                448,  # height
                448,  # width
                3,  # num_channels
            ],
        )
        aspect_ratio_ids_type = TensorType(
            DType.int64,
            shape=[
                1,
                1,
            ],  # batch_size, num_concurrent_media
        )
        aspect_ratio_mask_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                1,
                1,
                4,
            ],  # batch_size, num_concurrent_media, num_tiles
        )

        input_ids_type = TensorType(DType.int64, shape=[1, 14])  # patch_size
        # Same shapes.
        attention_mask_type = input_ids_type
        position_ids_type = input_ids_type
        cross_attention_mask_type = TensorType(DType.int64, [1, 14, 1, 4])

        blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = self.kv_manager.input_symbols()[
            0
        ]
        with Graph(
            "llama3-vision",
            input_types=[
                pixel_values_type,
                aspect_ratio_ids_type,
                aspect_ratio_mask_type,
                input_ids_type,
                attention_mask_type,
                cross_attention_mask_type,
                position_ids_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                is_cache_empty_type,
            ],
        ) as graph:
            model = ConditionalGenerator(
                pipeline_config=self.pipeline_config,
                vision_model=instantiate_vision_model(
                    self.pipeline_config, self.weights
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
                    pipeline_config=self.pipeline_config,
                    kv_params=self._get_kv_params(),
                    weights=self.weights,
                ),
            )

            (
                pixel_values,
                aspect_ratio_ids,
                aspect_ratio_mask,
                input_ids,
                attention_mask,
                cross_attention_mask,
                position_ids,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = graph.inputs

            # TODO: Call model.prepare_cross_attention_mask(...) here before
            # passing in the cross attention mask in the step below.
            # We will then have to call something like Tensor.from_numpy(...)
            # before the forward call below.
            outputs = model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                kv_cache_inputs=(
                    blocks,
                    cache_lengths,
                    lookup_table,
                    is_cache_empty,
                ),
            )

            loss, logits, past_key_values, hidden_states, attentions = outputs  # type: ignore
            graph.output(logits)  # type: ignore
            return graph

    def prepare_initial_token_inputs(
        self, context_batch: list[TextAndVisionContext]
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError("not yet implemented.")

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError("not yet implemented.")

    def execute(self, *model_inputs: Tensor) -> tuple[Tensor, ...]:
        return self.model.execute(*model_inputs, copy_inputs_to_device=False)[0]  # type: ignore

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
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,  # verify this.
            max_seq_len=self.pipeline_config.huggingface_config.text_config.max_position_embeddings,  # verify this.
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
            devices=[self.pipeline_config.device],
            session=session,
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
