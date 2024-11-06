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
import numpy as np
from max.pipelines import (
    PreTrainedTokenGeneratorTokenizer,
    PipelineConfig,
    WeightsFormat,
)
from max.pipelines.interfaces import TokenGeneratorRequest
from transformers import AutoTokenizer
from dataprocessing import TextContext, max_tokens_to_generate
from utils import tokenizer_from_gguf
from pathlib import Path


async def run_with_default_executor(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


# These aren't thread safe, but we don't want them running on the main
# thread. Guard them with an async lock for now.
_TOKENIZER_LOCK = asyncio.Lock()


class TextTokenizer(PreTrainedTokenGeneratorTokenizer[TextContext]):
    """Encapsulates creation of TextContext and specific token encode/decode logic.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        if config.weight_path is None and config.huggingface_repo_id is None:
            msg = (
                "a huggingface_repo_id or gguf compatible weights_path must be"
                " provided."
            )
            raise ValueError(msg)
        elif (
            config.weight_path is not None
            and config.huggingface_repo_id is None
        ):
            if config.weights_format == WeightsFormat.gguf:
                # Tokenizer_from_gguf will try to load the tokenizer.
                # However, this currently only supports llama based tokenizer.
                # TODO: AIPIPE-163 - Generalize or retire tokenizer_from_gguf.
                super().__init__(tokenizer_from_gguf(Path(config.weight_path)))
            else:
                msg = (
                    "a huggingface_repo_id or gguf compatible weights_path must"
                    " be provided."
                )
                raise ValueError(msg)

        elif (
            config.weight_path is not None
            and config.huggingface_repo_id is not None
        ):
            if config.weights_format == WeightsFormat.gguf:
                try:
                    super().__init__(
                        AutoTokenizer.from_pretrained(
                            config.huggingface_repo_id,
                            gguf_file=config.weight_path,
                        )
                    )
                except Exception:
                    super().__init__(
                        AutoTokenizer.from_pretrained(
                            config.huggingface_repo_id
                        )
                    )

            else:
                super().__init__(
                    AutoTokenizer.from_pretrained(config.huggingface_repo_id)
                )
        else:
            # This scenario, only arises, when the huggingface_repo_id is provided, and a weight path is not.
            super().__init__(
                AutoTokenizer.from_pretrained(config.huggingface_repo_id)
            )

    async def encode(self, prompt: str) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        async with _TOKENIZER_LOCK:
            encoded_prompt = await run_with_default_executor(
                self.delegate.encode, prompt
            )

        if len(encoded_prompt) >= self.config.max_length:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater than the"
                " configured max model context length of"
                f" {self.config.max_length}"
            )
            raise ValueError(msg)

        return encoded_prompt

    async def decode(
        self, context: TextContext, encoded: np.ndarray, **kwargs
    ) -> str:
        """Transformer a provided encoded token array, back into readable text.
        """

        return self.delegate.decode(encoded, **kwargs)

    async def new_context(self, request: TokenGeneratorRequest) -> TextContext:
        """Create a new TextContext object, leveraging necessary information like
        cache_seq_id and prompt from TokenGeneratorRequest."""
        encoded_prompt = await self.encode(request.prompt)

        max_gen_tokens = max_tokens_to_generate(
            len(encoded_prompt),
            self.config.max_length,
            request.max_new_tokens if request.max_new_tokens
            is not None else self.config.max_new_tokens,
        )
        context = TextContext(
            prompt=request.prompt,
            cache_seq_id=request.index,
            max_tokens=len(encoded_prompt) + max_gen_tokens,
        )
        context.append(np.array(encoded_prompt))
        return context
