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

from .config import InferenceConfig
from .context import ReplitContext

import transformers
import gguf
from typing import Optional
from max.engine import InferenceSession, Model
from max.pipelines import TokenGenerator
from max.graph.weights import GGUFWeights

from nn.kv_cache import KVCacheParams, load_kv_manager

from .model.hyperparameters import Hyperparameters
from .model.graph import _build_graph


class Replit(TokenGenerator):
    """The overall interface to the Replit model."""

    def __init__(self, config: InferenceConfig, **kwargs):
        self._config = config

        # Read in Hyperparameters.
        self._hyperparameters = Hyperparameters.load(config, **kwargs)

        # Load Device.
        self._device = self._config.device()

        # Get KV Cache Params.
        self._kv_params = KVCacheParams(
            dtype=self._hyperparameters.dtype,
            n_kv_heads=self._hyperparameters.n_kv_heads,
            head_dim=self._hyperparameters.head_dim,
            cache_strategy=self._config.cache_strategy,
        )

        # Load KV Cache Manager.
        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=self._config.max_cache_batch_size,
            max_seq_len=self._config.max_length,
            num_layers=self._hyperparameters.num_layers,
            device=self._device,
        )

        # Load Tokenizer from GGUF.
        if self._config.weight_path is None:
            raise ValueError(
                "no weight path provided for replit based gguf weights."
            )

        gguf_reader = gguf.GGUFReader(self._config.weight_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "replit/replit-code-v1_5-3b", padding_side="right"
        )

        # Load Weights from GGUF.
        self._weights = GGUFWeights(gguf_reader)

        # Load Model.
        session = InferenceSession(device=self._device)
        self._model = self._load_model(session)

    def _load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if serialized_path := self._config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():
                weights_registry[name] = tensor.data
            print("Loading serialized model from ", serialized_path, "...")
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            print("Building model...")
            graph = _build_graph(
                self._hyperparameters, self._weights, self._kv_params
            )
            print("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights
            )

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> ReplitContext:
        raise NotImplementedError("replit not yet implemented.")

    async def next_token(
        self, batch: dict[str, ReplitContext]
    ) -> dict[str, str]:
        raise NotImplementedError("replit not yet implemented.")

    async def release(self, context: ReplitContext):
        raise NotImplementedError("replit not yet implemented.")
