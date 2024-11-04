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
import numpy as np

from dataprocessing import (
    collate_batch,
    max_tokens_to_generate,
)
from max.driver import CPU, CUDA, Tensor
from transformers import AutoTokenizer
from typing import Any
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.pipelines import PreTrainedTokenGeneratorTokenizer
from nn.sampling import token_sampler
from max.graph.weights import SafetensorWeights
from max.pipelines.kv_cache import (
    KVCacheParams,
    load_kv_manager,
)
from .config import InferenceConfig
from .model import transformer
from .hyperparameters import Hyperparameters


@dataclass
class MistralContext:
    """The context for text generation using a Mistral model."""

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
        """Appends to the generated tokens and decoded output."""
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


class MistralTokenizer(PreTrainedTokenGeneratorTokenizer[MistralContext]):
    """Encapsulates Mistral specific token encode/decode logic."""

    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config
        super().__init__(
            AutoTokenizer.from_pretrained(config.repo_id, padding_side="right")
        )

    async def encode(self, prompt: str) -> np.ndarray:
        # Encodes a prompt using the tokenizer, raising a ValueError if the
        # prompt exceeds the configured maximum length.

        # Don't run compute-bound work on the main thread
        # however, it's not thread-safe, so make sure only one can
        # run at a time.
        # TODO: This should go on its own process or a thread on the model process.
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
        context: MistralContext,
        encoded: np.ndarray,
    ) -> str:
        return self.delegate.decode(encoded)

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> MistralContext:
        encoded_prompt = await self.encode(request.prompt)

        _max_tokens_to_generate = max_tokens_to_generate(
            len(encoded_prompt),
            self.config.max_length,
            request.max_new_tokens if request.max_new_tokens
            is not None else self.config.max_new_tokens,
        )
        context = MistralContext(
            prompt=request.prompt,
            cache_seq_id=request.index,
            max_tokens=len(encoded_prompt) + _max_tokens_to_generate,
        )
        context.append(np.array(encoded_prompt))
        return context


class MistralTokenGenerator(TokenGenerator[MistralContext]):
    """Token Generator for the Mistral model."""

    def __init__(self, config: InferenceConfig, eos: int):
        self.config = config
        self.eos = eos

        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)
        session = InferenceSession(device=self._device)

        self.model = Mistral(config, session=session)

        # logits are always float32 for now
        self._sampler = session.load(token_sampler(config.top_k, DType.float32))

    def next_token(
        self, batch: dict[str, MistralContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(
        self, req_to_context_dict: dict[str, MistralContext]
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

    def release(self, context: MistralContext):
        self.model.release(context)


class Mistral:
    """The overall interface to the Mistral model."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        session: InferenceSession | None = None,
    ):
        self.config = config
        assert config.weight_path is not None
        self.weights = SafetensorWeights([config.weight_path])
        self.params = _read_hyperparameters(config)

        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        self._kv_params = KVCacheParams(
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            dtype=self.params.dtype,
            cache_strategy=self.params.cache_strategy,
        )

        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=config.max_cache_batch_size,
            max_seq_len=config.max_length,
            num_layers=self.params.n_layers,
            device=self._device,
        )

        session = InferenceSession(device=self._device)

        self._model = self._load_model(session, config)

        self._n_heads = self.params.n_heads

    def export_mef(self, export_path):
        self._model._export_mef(export_path)

    def _mistral_graph(
        self,
    ) -> Graph:
        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        valid_lengths_type = TensorType(DType.uint32, shape=["batch_size"])

        kv_cache_args = self._kv_manager.input_symbols()

        with Graph(
            "mistral",
            input_types=[
                tokens_type,
                valid_lengths_type,
                *kv_cache_args,
            ],
        ) as graph:
            model = transformer(
                graph, self.params, self.weights, self._kv_params
            )
            tokens, valid_lengths, *kv_cache = graph.inputs
            logits = model(
                tokens,
                valid_lengths,
                kv_cache,
            )
            graph.output(logits)
            return graph

    def _load_model(
        self,
        session: InferenceSession,
        config: InferenceConfig,
    ) -> Model:
        # TODO: test serialized_path with Mistral and add option to load from serialized file.

        self._weights = SafetensorWeights([config.weight_path])
        print("Building model...")
        graph = self._mistral_graph()
        print("Compiling...")
        return session.load(
            graph, weights_registry=self.weights.allocated_weights
        )

    def release(self, context: MistralContext):
        self._kv_manager.release(context.cache_seq_id)

    def _execute(self, batch: dict[str, MistralContext]) -> Tensor:
        for context in batch.values():
            if context.cache_seq_id in self._kv_manager.slots_remaining:
                self._kv_manager.external_claim([context.cache_seq_id])

        context_batch = batch.values()
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get valid lengths: unpadded lengths of each token vector in the batch.
        batch_size = len(context_batch)
        unpadded_lengths = [ctx.seq_len for ctx in context_batch]
        valid_lengths = Tensor.from_numpy(np.array(unpadded_lengths, np.uint32))

        # Pad tokens and compute attention mask for the batch.
        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]

        next_tokens_batch, _ = collate_batch(
            tokens,
            batch_size=len(tokens),
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        # Grab kv_collection.
        kv_cache_tensors = self._kv_manager.fetch(cache_seq_ids)

        # Execute model.
        logits = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self._device),
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


def _read_hyperparameters(
    config: InferenceConfig,
) -> Hyperparameters:
    cache_strategy = config.cache_strategy
    seq_len = config.max_length

    return Hyperparameters(
        dtype=DType.bfloat16,
        quantization_encoding=config.quantization_encoding,
        seq_len=seq_len,
        cache_strategy=cache_strategy,
        has_dedicated_output_weights=True,
    )


async def release(self, context: MistralContext):
    await self._kv_manager.release(context.cache_seq_id)
