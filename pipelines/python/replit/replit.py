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

import asyncio
from typing import Optional

import gguf
import transformers
import numpy as np
from dataprocessing import (
    max_tokens_to_generate,
    collate_batch,
    causal_attention_mask_with_alibi,
)
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import GGUFWeights
from max.pipelines import TokenGenerator
from nn.kv_cache import KVCacheParams, load_kv_manager
from nn import argmax_sampler

from .config import InferenceConfig
from .context import ReplitContext
from .model.graph import _build_graph
from .model.hyperparameters import Hyperparameters


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


# These aren't thread safe, but we don't want them running on the main
# thread. Guard them with an async lock for now.
_TOKENIZER_LOCK = asyncio.Lock()
_ENGINE_LOCK = asyncio.Lock()


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
        session = InferenceSession(device=self._device)
        self._model = self._load_model(session)

        # Load Sampler.
        self._sampler = session.load(argmax_sampler(DType.float32))

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
                self._hyperparameters,
                self._weights,
                self._kv_params,
                self._kv_manager,
            )
            print("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights
            )

    async def _encode(self, prompt: str) -> list[int]:
        async with _TOKENIZER_LOCK:
            encoded_prompt = await run_with_default_executor(
                self._tokenizer.encode, prompt
            )
        if len(encoded_prompt) >= self._config.max_length:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater or equal to"
                " configured max model context length of"
                f" {self._config.max_length}."
            )
            raise ValueError(msg)

        return encoded_prompt

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> ReplitContext:
        encoded_prompt = await self._encode(prompt)

        _max_tokens_to_generate = max_tokens_to_generate(
            len(encoded_prompt),
            self._config.max_length,
            max_new_tokens if max_new_tokens
            is not None else self._config.max_new_tokens,
        )
        seq_id = await self._kv_manager.claim(n=1)
        context = ReplitContext(
            prompt=prompt,
            max_tokens=len(encoded_prompt) + _max_tokens_to_generate,
            cache_seq_id=seq_id[0],
        )
        context.append(np.array(encoded_prompt), prompt)
        return context

    def _execute(self, batch: dict[str, ReplitContext]) -> Tensor:
        """Executes the model and returns the raw results."""

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
            alibi_bias_max=self._hyperparameters.alibi_bias_max,
            n_heads=self._hyperparameters.n_heads,
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

    async def next_token(
        self, batch: dict[str, ReplitContext]
    ) -> dict[str, str]:
        res = {}

        # Don't run compute-bound work on the main thread
        # however, it's not thread-safe, so make sure only one can
        # run at a time.
        async with _ENGINE_LOCK:
            logits = await run_with_default_executor(self._execute, batch)
            (tokens,) = await run_with_default_executor(self._sampler, logits)
            tokens = tokens.to(CPU())

        next_tokens = dict(zip(batch, tokens.to_numpy()))
        for request_id, context in batch.items():
            next_token = next_tokens[request_id].astype(np.int64)
            decoded_token = self._tokenizer.decode(next_token)

            # Update context
            context.append(next_token.reshape(-1), decoded_token)

            # Mark completed requests by not including them in the response.
            if not context.is_done(self._tokenizer.eos_token_id):
                res[request_id] = decoded_token

        return res

    async def release(self, context: ReplitContext):
        await self._kv_manager.release(context.cache_seq_id)
