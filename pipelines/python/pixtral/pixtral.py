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
import logging

import numpy as np
from dataprocessing import causal_attention_mask_with_alibi, collate_batch
from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import (
    PipelineConfig,
    PipelineModel,
    TextContext,
    TokenGenerator,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    load_kv_manager,
)
from transformers import AutoProcessor

from .model.graph import _build_graph


class PixtralModel(TokenGenerator):
    """The overall interface to the Pixtral model."""

    def __init__(self, config: PipelineConfig, **kwargs):
        self._config = config

        # Load Device.
        self._device = self._config.device
        # session = InferenceSession(device=self._device)
        session = InferenceSession()

        # Get KV Cache Params.
        self._kv_params = KVCacheParams(
            dtype=self._config.dtype,
            n_kv_heads=self._config.huggingface_config.text_config.num_key_value_heads,
            head_dim=self._config.huggingface_config.text_config.head_dim,
            cache_strategy=self._config.cache_strategy,
        )

        # Load KV Cache Manager.
        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=self._config.max_cache_batch_size,
            max_seq_len=self._config.huggingface_config.max_seq_len,
            num_layers=self._config.huggingface_config.text_config.num_hidden_layers,
            devices=[self._device],
            session=session,
        )

        # Load Processor from HuggingFace.
        self._processor = AutoProcessor.from_pretrained(
            "mistral-community/pixtral-12b"
        )

        # Load Weights from SafeTensors.
        if self._config.weight_path is None:
            raise ValueError(
                "no weight path provided for mistral based safetensor weights."
            )

        self._weights = SafetensorWeights(self._config.weight_path)

        # Load Model.
        self._model = self._load_model(session)

        # Load Sampler.
        # self._sampler = session.load(
        #    token_sampler(self._config.top_k, DType.float32)
        # )

    def _load_model(
        self,
        session: InferenceSession,
    ):
        if serialized_path := self._config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():  # type: ignore
                weights_registry[name] = tensor.data
            logging.info(
                "Loading serialized model from ", serialized_path, "..."
            )
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            logging.info("Building model...")
            graph = _build_graph(
                self._config,
                self._weights,
                self._kv_params,
                self._kv_manager,
            )
            # logging.info("Compiling...")
