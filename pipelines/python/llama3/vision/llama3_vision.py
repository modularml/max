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

from dataclasses import dataclass

from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType, ops
from max.graph.weights import SafetensorWeights
from nn import Linear

from .conditional_generator import ConditionalGenerator
from .config import InferenceConfig
from .hyperparameters import TextHyperparameters, VisionHyperparameters
from .language_model import instantiate_language_model
from .vision_model import instantiate_vision_model


# TODO(AIPIPE-142): Implement this.
@dataclass
class Llama3VisionContext:
    """The context for text generation using a Llama3.2 vision model."""


# TODO: Some parts may be consolidated under the parent Llama 3 pipeline interface.
class Llama3Vision:
    """The Llama3.2 vision model."""

    def __init__(
        self,
        config: InferenceConfig,
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

        session = InferenceSession(devices=[self._device])

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

        with Graph(
            "llama3-vision",
            input_types=[
                pixel_values_type,
                aspect_ratio_ids_type,
                aspect_ratio_mask_type,
                attention_mask_type,
                input_ids_type,
                attention_mask_type,
                position_ids_type,
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
                    self.text_params, self.weights
                ),
            )

            keys = [
                "pixel_values",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
                "input_ids",
                "attention_mask",
                "position_ids",
            ]
            kwargs = dict(zip(keys, graph.inputs))

            # TODO: Figure out how we want to pass these into the graph, either
            #       hardcoded or as graph inputs.
            # cross_attention_mask: shape=[1, 1, 14, 4100], dtype=torch.bfloat16
            # past_key_values: value=DynamicCache()
            # inputs_embeds: value=None
            # cache_position: shape=[14], dtype=torch.int64
            outputs = model(**kwargs)  # type: ignore

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
            graph, weights_registry=self.weights.allocated_weights  # type: ignore
        )
        print("Done!")
        return res

    def _execute(
        self, req_to_context_dict: dict[str, Llama3VisionContext]
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
