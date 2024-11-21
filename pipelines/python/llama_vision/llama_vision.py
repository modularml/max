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

from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType, ops
from max.graph.weights import SafetensorWeights
from max.pipelines import TextAndVisionContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from nn import Linear

from .conditional_generator import ConditionalGenerator
from .config import InferenceConfig
from .hyperparameters import TextHyperparameters, VisionHyperparameters
from .language_model import instantiate_language_model
from .vision_model import instantiate_vision_model


# TODO: These are configured for text only model. What about vision model?
def load_llama_vision_and_kv_manager(
    config: InferenceConfig, session: InferenceSession | None = None
) -> tuple[LlamaVision, KVCacheManager]:
    _, text_params = _read_hyperparameters(config)
    # Initialize kv cache params and manager
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=text_params.num_attention_heads,
        head_dim=text_params.hidden_size // text_params.num_attention_heads,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )

    # TODO: Duplicated code for now. Remove and consolidate somewhere else.
    curr_device = CPU(
        config.device_spec.id
    ) if config.device_spec.device_type == "cpu" else CUDA(
        config.device_spec.id
    )

    if session is None:
        session = InferenceSession(devices=[curr_device])

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=1,  # verify this.
        max_seq_len=text_params.max_position_embeddings,  # verify this.
        num_layers=text_params.num_hidden_layers,
        devices=[curr_device],
        session=session,
    )
    model = LlamaVision(
        config=config,
        kv_manager=kv_manager,
        session=session,
    )

    return model, kv_manager


# TODO: Some parts may be consolidated under the parent Llama 3 pipeline interface.
class LlamaVision:
    """The Llama3.2 vision model."""

    def __init__(
        self,
        config: InferenceConfig,
        kv_manager: KVCacheManager,
        *,
        session: InferenceSession | None = None,
    ):
        self.config = config

        # Llama 3.2 vision model always takes in multiple safetensors, so we assert
        # here to check if that's always true.
        assert config.weight_path is not None and isinstance(
            config.weight_path, list
        )
        assert all(isinstance(item, str) for item in config.weight_path)
        self.weights = SafetensorWeights(config.weight_path)  # type: ignore
        self.vision_params, self.text_params = _read_hyperparameters(config)

        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        if session is None:
            session = InferenceSession(devices=[self._device])

        self._kv_manager = kv_manager
        self._kv_params = self._kv_manager.params

        self._model = self._load_model(session)

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
            DType.bfloat16,
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
            DType.bfloat16,
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

        blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
            self._kv_manager.input_symbols()
        )
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
                text_params=self.text_params,
                vision_params=self.vision_params,
                vision_model=instantiate_vision_model(
                    self.vision_params, self.weights
                ),
                multi_modal_projector=Linear(
                    self.weights.multi_modal_projector.weight.allocate(
                        DType.bfloat16,
                        [
                            self.text_params.hidden_size,
                            self.vision_params.vision_output_dim,
                        ],
                    ),
                    self.weights.multi_modal_projector.bias.allocate(
                        DType.bfloat16,
                        [self.text_params.hidden_size],
                    ),
                ),
                language_model=instantiate_language_model(
                    kv_params=self._kv_params,
                    params=self.text_params,
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

    def _load_model(
        self,
        session: InferenceSession,
    ) -> Model | None:
        print("Building model...")
        graph = self._llama3_vision_graph()
        print("Compiling...")
        res = session.load(
            graph,
            weights_registry=self.weights.allocated_weights,  # type: ignore
        )
        print("Done!")
        return res

    def _execute(
        self, req_to_context_dict: dict[str, TextAndVisionContext]
    ) -> Tensor:
        raise NotImplementedError("Not implemented yet")


def _read_hyperparameters(
    config: InferenceConfig,
) -> tuple[VisionHyperparameters, TextHyperparameters]:
    return (
        VisionHyperparameters(
            dtype=DType.bfloat16,
            quantization_encoding=config.quantization_encoding,  # type: ignore
        ),
        TextHyperparameters(
            dtype=DType.bfloat16,
            quantization_encoding=config.quantization_encoding,  # type: ignore
        ),
    )
