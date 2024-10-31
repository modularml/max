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

import numpy as np
from dataprocessing import max_tokens_to_generate
from max.driver import CPU, CUDA
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import PreTrainedTokenGeneratorTokenizer
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from nn.sampling import token_sampler

from utils import tokenizer_from_gguf

from .config import InferenceConfig
from .llama3 import Llama3


@dataclass
class Llama3Context:
    """The context for text generation using a Llama 3 model."""

    prompt: str
    """Input prompt string prior to tokenization."""

    max_tokens: int
    """The maximum number of tokens to generate, including the prompt."""

    cache_seq_id: int
    """Sequence id to tell the KV cache manager which cache block this owns."""

    next_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    """A (seq_len,) vector of the input tokens for this iteration."""

    tokens: list[int] = field(default_factory=list)
    """Tokens generated so far."""

    def append(self, token_ids: np.ndarray) -> None:
        """Appends to the generated tokens"""
        assert len(token_ids.shape) == 1
        self.next_tokens = token_ids
        self.tokens.extend(token_ids)

    def is_done(self, eos: int) -> bool:
        """Returns true if token gen for this context completed, else false."""
        return self.tokens[-1] == eos or len(self.tokens) > self.max_tokens

    @property
    def seq_len(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        return self.next_tokens.shape[-1]


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


# These aren't thread safe, but we don't want them running on the main
# thread. Guard them with an async lock for now.
_TOKENIZER_LOCK = asyncio.Lock()


class Llama3Tokenizer(PreTrainedTokenGeneratorTokenizer[Llama3Context]):
    """Encapsulates Llama3 specific token encode/decode logic."""

    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config
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
        if len(encoded_prompt) >= self.config.max_length:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater or equal to"
                " configured max model context length of"
                f" {self.config.max_length}."
            )
            raise ValueError(msg)

        return encoded_prompt

    async def decode(
        self,
        context: Llama3Context,
        encoded: np.ndarray,
    ) -> str:
        return self.delegate.decode(encoded)

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> Llama3Context:
        encoded_prompt = await self.encode(request.prompt)

        _max_tokens_to_generate = max_tokens_to_generate(
            len(encoded_prompt),
            self.config.max_length,
            request.max_new_tokens if request.max_new_tokens
            is not None else self.config.max_new_tokens,
        )
        context = Llama3Context(
            prompt=request.prompt,
            cache_seq_id=request.index,
            max_tokens=len(encoded_prompt) + _max_tokens_to_generate,
        )
        context.append(np.array(encoded_prompt))
        return context


class Llama3TokenGenerator(TokenGenerator[Llama3Context]):
    """Token Generator for the Llama 3 model."""

    def __init__(self, config: InferenceConfig, eos: int, vocab_size: int):
        self.config = config
        self.eos = eos

        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)
        session = InferenceSession(device=self._device)

        self.model = Llama3(config, session=session, vocab_size=vocab_size)

        # logits are always float32 for now
        self._sampler = session.load(token_sampler(config.top_k, DType.float32))

        if export_path := config.save_to_serialized_model_path:
            print(f"Exporting serialized model to {export_path}...")
            self.model.export_mef(export_path)

    def next_token(
        self, batch: dict[str, Llama3Context], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> dict[str, Any]:
        res = {}
        logits = self.model._execute(req_to_context_dict)
        tokens = self._sampler(logits)[0]
        tokens = tokens.to(CPU())

        next_tokens = dict(zip(req_to_context_dict, tokens.to_numpy()))
        for request_id, context in req_to_context_dict.items():
            next_token = next_tokens[request_id].astype(np.int64)

            # Update context
            context.append(next_token.reshape(-1))

            # Mark completed requests by not including them in the response.
            if not context.is_done(self.eos):
                res[request_id] = next_token

        return res

    def release(self, context: Llama3Context):
        self.model.release(context)
