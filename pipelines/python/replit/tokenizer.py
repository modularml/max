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
import asyncio
import gguf
import numpy as np
from dataprocessing import max_tokens_to_generate

from max.pipelines import PreTrainedTokenGeneratorTokenizer
from max.pipelines.interfaces import TokenGeneratorRequest
from transformers import AutoTokenizer


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


def gguf_reader_and_params(config: InferenceConfig):
    assert config.weight_path is not None
    reader = gguf.GGUFReader(config.weight_path)
    return reader


_TOKENIZER_LOCK = asyncio.Lock()


class ReplitTokenizer(PreTrainedTokenGeneratorTokenizer[ReplitContext]):
    """Encapsulates Llama3 specific token encode/decode logic."""

    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config
        super().__init__(
            AutoTokenizer.from_pretrained("modularai/replit-code-1.5")
        )

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
        context: ReplitContext,
        encoded: np.ndarray,
    ) -> str:
        return self.delegate.decode(encoded)

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> ReplitContext:
        encoded_prompt = await self.encode(request.prompt)

        _max_tokens_to_generate = max_tokens_to_generate(
            len(encoded_prompt),
            self.config.max_length,
            request.max_new_tokens if request.max_new_tokens
            is not None else self.config.max_new_tokens,
        )
        context = ReplitContext(
            prompt=request.prompt,
            cache_seq_id=request.index,
            max_tokens=len(encoded_prompt) + _max_tokens_to_generate,
        )
        context.append(np.array(encoded_prompt))
        return context
