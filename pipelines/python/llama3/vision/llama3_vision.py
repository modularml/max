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
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights

from .config import InferenceConfig
from .hyperparameters import VisionHyperparameters
from .model import instantiate_vision_model


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
        self.weights = SafetensorWeights(config.weight_path)
        self.params = _read_hyperparameters(config)

        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        session = InferenceSession(device=self._device)

        self._model = self._load_model(session)

    def export_mef(self, export_path):
        self._model._export_mef(export_path)

    def _llama3_vision_graph(
        self,
    ) -> Graph:
        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        valid_lengths_type = TensorType(DType.uint32, shape=["batch_size"])

        with Graph(
            "llama3-vision",
            input_types=[
                tokens_type,
                valid_lengths_type,
            ],
        ) as graph:
            # TODO: Implement this.
            model = instantiate_vision_model(
                graph,
                self.params,
                self.weights,
            )
            # tokens, valid_lengths = graph.inputs
            # logits = model(
            #     tokens,
            #     valid_lengths,
            # )
            # graph.output(logits)
            return graph

    def _load_model(
        self,
        session: InferenceSession,
    ) -> Model | None:
        print("Building model...")
        graph = self._llama3_vision_graph()
        # print("Compiling...")
        # TODO: Stubbing out for now.
        # return session.load(
        #     graph, weights_registry=self.weights.allocated_weights
        # )
        return None

    def _execute(
        self, req_to_context_dict: dict[str, Llama3VisionContext]
    ) -> Tensor:
        raise NotImplementedError("Not implemented yet")


def _read_hyperparameters(
    config: InferenceConfig,
) -> VisionHyperparameters:
    return VisionHyperparameters(
        dtype=DType.bfloat16,
        quantization_encoding=config.quantization_encoding,
    )
