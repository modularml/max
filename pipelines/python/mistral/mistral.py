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
from typing import Any

import numpy as np
import transformers
from dataprocessing import causal_attention_mask_with_alibi, collate_batch
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig, TextContext, TokenGenerator
from max.pipelines.kv_cache import KVCacheParams, load_kv_manager
from max.pipelines.response import TextResponse
from max.pipelines.sampling import token_sampler
from transformers import AutoTokenizer

from .model.graph import _build_graph

logger = logging.getLogger(__name__)


class Mistral(TokenGenerator):
    """The overall interface to the Mistral model."""

    def __init__(self, config: PipelineConfig, **kwargs):
        self._config = config

        # Load Device.
        self._device = self._config.device
        session = InferenceSession(devices=[self._device])

        # Get KV Cache Params.
        self._kv_params = KVCacheParams(
            dtype=self._config.dtype,
            n_kv_heads=self._config.huggingface_config.num_key_value_heads,
            head_dim=self._config.huggingface_config.head_dim,
            cache_strategy=self._config.cache_strategy,
        )

        # Load KV Cache Manager.
        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=self._config.max_cache_batch_size,
            max_seq_len=self._config.huggingface_config.max_seq_len,
            num_layers=self._config.huggingface_config.num_hidden_layers,
            devices=[self._device],
            session=session,
        )

        # Load Tokenizer from HuggingFace.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            "mistralai/Mistral-Nemo-Instruct-2407", padding_side="right"
        )

        # Load Weights from GGUF.
        if self._config.weight_path is None:
            raise ValueError(
                "no weight path provided for mistral based safetensor weights."
            )

        self._weights = SafetensorWeights([self._config.weight_path])  # type: ignore

        # Load Model.
        self._model = self._load_model(session)

        # Load Sampler.
        self._sampler = session.load(
            token_sampler(self._config.top_k, DType.float32)
        )

    def _load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if serialized_path := self._config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():  # type:ignore
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
            logging.info("Compiling...")
            return session.load(
                graph,
                weights_registry=self._weights.allocated_weights,  # type:ignore
            )

    def _execute(self, batch: dict[str, TextContext]) -> Tensor:
        for context in batch.values():
            if context.cache_seq_id in self._kv_manager.slots_remaining:
                self._kv_manager.external_claim([context.cache_seq_id])

        context_batch = batch.values()
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get input_row_offset: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offset = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch], dtype=np.uint32
            )
        )

        # Pad tokens and compute attention mask for the batch.
        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        next_tokens_batch = np.concatenate(tokens)

        # Grab kv_collection.
        kv_cache_tensors = self._kv_manager.fetch(cache_seq_ids)[0]

        # Execute model.
        logits = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self._device),
            input_row_offset.to(self._device),
            *kv_cache_tensors,
            copy_inputs_to_device=False,
        )[0]

        self._kv_manager.step(
            valid_lengths={
                ctx.cache_seq_id: ctx.seq_len for ctx in context_batch
            }
        )

        return logits  # type: ignore

    def next_token(
        self, batch: dict[str, TextContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(self, batch: dict[str, TextContext]) -> dict[str, TextResponse]:
        res = {}
        logits = self._execute(batch)

        # We use dummy_prev_tokens_input to match the signature of the token_sampler
        # graph, which also concats the new tokens with existing tokens
        # in engine-level multistep execution.
        dummy_prev_tokens_input = Tensor.from_numpy(
            np.zeros((len(batch), 0), dtype=np.int64)
        ).to(self._device)

        tokens = self._sampler(logits, dummy_prev_tokens_input)[0]

        tokens = tokens.to(CPU())  # type: ignore

        next_tokens = dict(zip(batch, tokens.to_numpy()))
        for request_id, context in batch.items():
            next_token = next_tokens[request_id].astype(np.int64)

            # Update context
            context.update(new_tokens=next_token.reshape(-1))

            # Mark completed requests by not including them in the response.
            if not context.is_done(self._tokenizer.eos_token_id):
                res[request_id] = TextResponse(next_token=next_token)

        return res

    def release(self, context: TextContext):
        self._kv_manager.release(context.cache_seq_id)
