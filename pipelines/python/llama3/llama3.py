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
from typing import Any, Optional, Union

import gguf
import numpy as np
from dataprocessing import (
    batch_padded_tokens_and_mask,
    collate_batch,
    max_tokens_to_generate,
)
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, Graph, TensorType
from max.graph.weights import GGUFWeights
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.serve.pipelines.llm import PreTrainedTokenGeneratorTokenizer
from nn.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from nn.sampling import token_sampler
from tokenizers import Tokenizer

from utils import gguf_utils, tokenizer_from_gguf

from .config import InferenceConfig, SupportedVersions
from .gguf import transformer
from .model.hyperparameters import Hyperparameters


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


def gguf_reader_and_params(config: InferenceConfig):
    assert config.weight_path is not None
    reader = gguf.GGUFReader(config.weight_path)
    params = _read_hyperparameters(config, reader)
    return reader, params


class Llama3Tokenizer(PreTrainedTokenGeneratorTokenizer[Llama3Context]):
    """Encapsulates Llama3 specific token encode/decode logic."""

    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config
        self.reader, self.params = gguf_reader_and_params(config)
        super().__init__(tokenizer_from_gguf(self.reader))

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


class Llama3(TokenGenerator[Llama3Context]):
    """The overall interface to the Llama 3 model."""

    def __init__(self, config: InferenceConfig, eos: int, vocab_size: int):
        self.config = config
        self.eos = eos

        assert config.weight_path is not None
        self.reader, self.params = gguf_reader_and_params(config)

        # Work around for older Llama 1/2 GGUFs, where the vocab size may be -1.
        # See https://github.com/ggerganov/llama.cpp/pull/4258.
        if self.params.vocab_size < 0:
            self.params.vocab_size = vocab_size

        dtype = (
            DType.float32 if self.params.quantization_encoding
            is not None else self.params.dtype
        )
        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        self._kv_params = KVCacheParams(
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            dtype=dtype,
            cache_strategy=self.config.cache_strategy,
        )

        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=config.max_cache_batch_size,
            max_seq_len=config.max_length,
            num_layers=self.params.n_layers,
            device=self._device,
        )

        session = InferenceSession(device=self._device)

        self._model = self._load_model(
            session, config, self.params, self.reader
        )
        # logits are always float32 for now
        self._sampler = session.load(token_sampler(config.top_k, DType.float32))

        if export_path := config.save_to_serialized_model_path:
            print(f"Exporting serialized model to {export_path}...")
            self._model._export_mef(export_path)

        self._n_heads = self.params.n_heads

    def _llama_graph_opaque(
        self,
        weights: GGUFWeights,
    ) -> Graph:
        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        valid_lengths_type = TensorType(DType.uint32, shape=["batch_size"])

        kv_cache_args = self._kv_manager.input_symbols()

        with Graph(
            "llama3",
            input_types=[
                tokens_type,
                valid_lengths_type,
                *kv_cache_args,
            ],
        ) as graph:
            model = transformer(
                graph, self.config, self.params, weights, self._kv_params
            )
            tokens, valid_lengths, *kv_cache = graph.inputs
            logits = model(
                tokens,
                valid_lengths,
                kv_cache,
            )
            graph.output(logits)
            return graph

    def _llama_graph(
        self,
        weights: GGUFWeights,
    ) -> Graph:
        if self.config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._llama_graph_opaque(weights)

        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )

        cache_type = BufferType(
            DType.float32,
            shape=[
                self.params.seq_len,
                self.params.n_layers,
                "max_batch_size",
                self.params.n_kv_heads,
                self.params.head_dim,
            ],
        )
        start_pos_type = TensorType(DType.int64, shape=[])

        with Graph(
            "llama3",
            input_types=[
                tokens_type,
                attn_mask_type,
                cache_type,
                cache_type,
                start_pos_type,
            ],
        ) as graph:
            model = transformer(
                graph, self.config, self.params, weights, self._kv_params
            )
            tokens, attention_mask, k_cache, v_cache, start_pos = graph.inputs
            logits, end_pos = model(
                tokens,
                attention_mask.cast(self.params.mask_dtype),
                k_cache,
                v_cache,
                start_pos,
            )
            graph.output(logits[:, -1], end_pos)
            return graph

    def _load_model(
        self,
        session: InferenceSession,
        config: InferenceConfig,
        params: Hyperparameters,
        reader: gguf.GGUFReader,
    ) -> Model:
        self._weights = GGUFWeights(reader)
        if serialized_path := config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():
                weights_registry[name] = tensor.data
            print("Loading serialized model from", serialized_path, "...")
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            print("Building model...")
            graph = self._llama_graph(self._weights)
            print("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights
            )

    def next_token(
        self, batch: dict[str, Llama3Context], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> dict[str, Any]:
        res = {}
        logits = self._execute(req_to_context_dict)
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
        self._kv_manager.release(context.cache_seq_id)

    def _execute_opaque(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> Tensor:
        context_batch = req_to_context_dict.values()
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

    def _execute(self, req_to_context_dict: dict[str, Llama3Context]) -> Tensor:
        """Executes the model and returns the raw results."""
        for context in req_to_context_dict.values():
            if context.cache_seq_id in self._kv_manager.slots_remaining:
                self._kv_manager.external_claim([context.cache_seq_id])

        if self.config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._execute_opaque(req_to_context_dict)

        context_batch = req_to_context_dict.values()
        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]
        tokens = [ctx.next_tokens for ctx in context_batch]
        batch_size = len(context_batch)

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self._kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(req_to_context_dict)
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=tokens,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        keys, values, seq_len, _ = self._kv_manager.fetch(cache_seq_ids)

        # Execute model.
        logits, end_pos = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self._device),
            Tensor.from_numpy(attn_mask).to(self._device),
            keys,
            values,
            seq_len,
        )

        end_pos = end_pos.to(CPU()).item()

        self._kv_manager.step(
            valid_lengths={
                ctx.cache_seq_id: ctx.seq_len for ctx in context_batch
            }
        )

        return logits


def _read_hyperparameters(
    config: InferenceConfig, reader: gguf.GGUFReader
) -> Hyperparameters:
    key_names = {
        "n_layers": "llama.block_count",
        "n_heads": "llama.attention.head_count",
        "n_kv_heads": "llama.attention.head_count_kv",
        "vocab_size": "llama.vocab_size",
        "hidden_dim": "llama.embedding_length",
        "rope_theta": "llama.rope.freq_base",
        "layer_norm_rms_epsilon": "llama.attention.layer_norm_rms_epsilon",
    }

    configured_params = {
        name: value
        for name, key in key_names.items()
        if (value := gguf_utils.read_number(reader, key)) is not None
    }

    # The feed forward length doesn't appear in the pretrained llama checkpoint
    # fields. Obtain the value from the shape of the projection weight.
    tensor = next(
        filter(lambda t: t.name == "blk.0.ffn_down.weight", reader.tensors)
    )
    feed_forward_length = tensor.shape[0]

    seq_len = 128_000 if config.version == SupportedVersions.llama3_1 else 8_000
    if config.max_length > seq_len:
        print(
            "Warning: `max_length` is more than the supported context size"
            f"`max_length` is now set to {seq_len}"
        )
        config.max_length = seq_len
    else:
        seq_len = config.max_length

    has_dedicated_output_weights = any(
        tensor.name == "output.weight" for tensor in reader.tensors
    )

    return Hyperparameters(
        dtype=config.quantization_encoding.dtype,
        quantization_encoding=config.quantization_encoding.quantization_encoding,
        feed_forward_length=feed_forward_length,
        seq_len=seq_len,
        has_dedicated_output_weights=has_dedicated_output_weights,
        **configured_params,
    )
