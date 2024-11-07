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

import gguf
import numpy as np
import transformers
from dataprocessing import (
    TextContext,
    causal_attention_mask_with_alibi,
    collate_batch,
)
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import GGUFWeights
from max.pipelines import PipelineConfig, TokenGenerator
from max.pipelines.kv_cache import KVCacheParams, load_kv_manager
from nn import token_sampler

from .model.graph import _build_graph

logger = logging.getLogger(__name__)


class Replit(TokenGenerator):
    """The overall interface to the Replit model."""

    def __init__(self, config: PipelineConfig, **kwargs):
        self._config = config

        # Load Device.
        self._device = self._config.device
        session = InferenceSession(device=self._device)

        # Get KV Cache Params.
        self._kv_params = KVCacheParams(
            dtype=self._config.dtype,
            n_kv_heads=self._config.huggingface_config.attn_config[
                "kv_n_heads"
            ],
            head_dim=self._config.huggingface_config.d_model
            // self._config.huggingface_config.n_heads,
            cache_strategy=self._config.cache_strategy,
        )

        # Load KV Cache Manager.
        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=self._config.max_cache_batch_size,
            max_seq_len=self._config.huggingface_config.max_seq_len,
            num_layers=self._config.huggingface_config.n_layers,
            device=self._device,
            session=session,
        )

        # Load Tokenizer from HuggingFace.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            "modularai/replit-code-1.5", padding_side="right"
        )

        # Load Weights from GGUF.
        if self._config.weight_path is None:
            raise ValueError(
                "no weight path provided for replit based gguf weights."
            )

        gguf_reader = gguf.GGUFReader(self._config.weight_path)
        self._weights = GGUFWeights(gguf_reader)

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
            for name, tensor in self._weights._tensors.items():
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
                graph, weights_registry=self._weights.allocated_weights
            )

    def _execute(self, batch: dict[str, TextContext]) -> Tensor:
        """Executes the model and returns the raw results."""
        for context in batch.values():
            if context.cache_seq_id in self._kv_manager.slots_remaining:
                self._kv_manager.external_claim([context.cache_seq_id])

        context_batch = batch.values()
        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get valid lengths: unpadded lengths of each token vector in the batch.
        batch_size = len(context_batch)
        unpadded_lengths = [ctx.seq_len for ctx in context_batch]
        valid_lengths = Tensor.from_numpy(np.array(unpadded_lengths, np.uint32))

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self._kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(batch)
        next_tokens_batch, _ = collate_batch(
            tokens,
            batch_size=len(tokens),
            pad_to_multiple_of=self._config.pad_to_multiple_of,
        )
        attention_mask = causal_attention_mask_with_alibi(
            original_start_pos=start_pos,
            original_seq_len=[len(t) for t in tokens],
            pad_to_multiple_of=self._config.pad_to_multiple_of,
            alibi_bias_max=self._config.huggingface_config.attn_config[
                "alibi_bias_max"
            ],
            n_heads=self._config.huggingface_config.n_heads,
        )

        # Grab kv_collection.
        kv_cache_tensors = self._kv_manager.fetch(cache_seq_ids)

        # Execute model.
        logits = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self._device),
            Tensor.from_numpy(attention_mask).to(self._device),
            valid_lengths.to(self._device),
            *kv_cache_tensors,
            copy_inputs_to_device=False,
        )[0]

        self._kv_manager.step(
            valid_lengths={
                ctx.cache_seq_id: ctx.seq_len for ctx in context_batch
            }
        )

        return logits

    def next_token(
        self, batch: dict[str, TextContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(self, batch: dict[str, TextContext]) -> dict[str, str]:
        res = {}
        logits = self._execute(batch)
        tokens = self._sampler(logits)[0]
        tokens = tokens.to(CPU())

        next_tokens = dict(zip(batch, tokens.to_numpy()))
        for request_id, context in batch.items():
            next_token = next_tokens[request_id].astype(np.int64)

            # Update context
            context.append(next_token.reshape(-1))

            # Mark completed requests by not including them in the response.
            if not context.is_done(self._tokenizer.eos_token_id):
                res[request_id] = next_token

        return res

    def release(self, context: TextContext):
        self._kv_manager.release(context.cache_seq_id)
