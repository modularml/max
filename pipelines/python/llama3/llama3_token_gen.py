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

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gguf
import numpy as np
from dataprocessing import TextContext, max_tokens_to_generate
from max.driver import CPU, CUDA
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import PipelineConfig, PreTrainedTokenGeneratorTokenizer
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.pipelines.kv_cache import KVCacheParams, load_kv_manager
from nn.sampling import token_sampler

from utils import tokenizer_from_gguf

from .llama3 import load_llama3_and_kv_manager, _read_hyperparameters


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


# These aren't thread safe, but we don't want them running on the main
# thread. Guard them with an async lock for now.
_TOKENIZER_LOCK = asyncio.Lock()


class Llama3Tokenizer(PreTrainedTokenGeneratorTokenizer[TextContext]):
    """Encapsulates Llama3 specific token encode/decode logic."""

    def __init__(
        self,
        config: PipelineConfig,
    ):
        self.config = config
        reader = gguf.GGUFReader(config.weight_path)
        self._hyperparameters = _read_hyperparameters(config, reader)
        assert config.weight_path is not None
        super().__init__(tokenizer_from_gguf(Path(config.weight_path)))

    async def encode(self, prompt: str) -> np.ndarray:
        # Encodes a prompt using the tokenizer, raising a ValueError if the
        # prompt exceeds the configured maximum length.

        # Don't run compute-bound work on the main thread
        # however, it's not thread-safe, so make sure only one can
        # run at a time.
        # TODO: This should go on its own process or a thread on the model process.
        assert self.delegate
        async with _TOKENIZER_LOCK:
            encoded_prompt = await run_with_default_executor(
                self.delegate.encode, prompt
            )
        if len(encoded_prompt) >= self._hyperparameters.seq_len:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater or equal to"
                " configured max model context length of"
                f" {self._hyperparameters.seq_len}."
            )
            raise ValueError(msg)

        return encoded_prompt

    async def decode(
        self,
        context: TextContext,
        encoded: np.ndarray,
    ) -> str:
        return self.delegate.decode(encoded)

    async def new_context(self, request: TokenGeneratorRequest) -> TextContext:
        encoded_prompt = await self.encode(request.prompt)

        _max_tokens_to_generate = max_tokens_to_generate(
            len(encoded_prompt),
            self._hyperparameters.seq_len,
            request.max_new_tokens if request.max_new_tokens
            is not None else self.config.max_new_tokens,
        )
        context = TextContext(
            prompt=request.prompt,
            cache_seq_id=request.index,
            max_tokens=len(encoded_prompt) + _max_tokens_to_generate,
        )
        context.append(np.array(encoded_prompt))
        return context


class Llama3TokenGenerator(TokenGenerator[TextContext]):
    """Token Generator for the Llama 3 model."""

    def __init__(self, config: PipelineConfig, eos: int, vocab_size: int):
        self.config = config
        self.eos = eos

        self._device = config.device
        session = InferenceSession(device=self._device)

        self.model, self._kv_manager = load_llama3_and_kv_manager(
            config,
            session,
            vocab_size=vocab_size,
        )

        # Logits are always float32 for now
        self._sampler = session.load(token_sampler(config.top_k, DType.float32))

        if export_path := config.save_to_serialized_model_path:
            print(f"Exporting serialized model to {export_path}...")
            self.model.export_mef(export_path)

    def next_token(
        self, batch: dict[str, TextContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        # Flatten our batch for consistent indexing
        context_batch = list(batch.values())

        # Claim cache rows for our batch
        for context in context_batch:
            if context.cache_seq_id in self._kv_manager.slots_remaining:
                self._kv_manager.external_claim([context.cache_seq_id])

        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]

        # Prepare inputs for the first token in multistep execution
        model_inputs = self.model._prepare_initial_token_inputs(context_batch)
        kv_cache_inputs = self._kv_manager.fetch(cache_seq_ids)

        # Multistep execution loop
        generated_tokens = []
        curr_step_inputs = model_inputs
        for i in range(num_steps):
            # Execute the model and get next tokens
            logits = self.model._execute(*curr_step_inputs, *kv_cache_inputs)
            new_tokens = self._sampler(logits)[0]

            generated_tokens.append(new_tokens)

            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                break

            # Prepare inputs for the next token in multistep execution
            kv_cache_inputs = self._kv_manager.increment_cache_lengths(
                kv_cache_inputs,
                curr_step_inputs,
            )
            curr_step_inputs = self.model._prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )

        # Actually update the cache lengths in our kv_cache manager
        self._kv_manager.step(
            valid_lengths={
                ctx.cache_seq_id: ctx.seq_len + num_steps - 1
                for ctx in context_batch
            }
        )

        # Do the copy to host for each token generated.
        generated_tokens = list(
            [g.to(CPU()).to_numpy() for g in generated_tokens]
        )

        # Prepare the response, pruning away completed requests as we go.
        res: list[dict[str, Any]] = []
        is_done = {r: False for r in batch.keys()}
        for i in range(num_steps):
            step_res = {}
            next_tokens = dict(zip(batch, generated_tokens[i]))
            for request_id, context in batch.items():
                if is_done[request_id]:
                    continue

                next_token = next_tokens[request_id].astype(np.int64)

                # Update context
                context.append(next_token.reshape(-1))

                # Mark completed requests by not including them in the response.
                if not context.is_done(self.eos):
                    step_res[request_id] = next_token
                else:
                    is_done[request_id] = True

            res.append(step_res)

        return res

    def release(self, context: TextContext):
        self._kv_manager.release(context.cache_seq_id)
