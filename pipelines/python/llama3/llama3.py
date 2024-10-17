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
from typing import Optional

import gguf
import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, Graph, TensorType, ops
from max.graph.weights import GGUFWeights
from nn.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from tokenizers import Tokenizer

from utils import gguf_utils, tokenizer_from_gguf

from .collate_batch import batch_padded_tokens_and_mask
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

    decoded: str = ""
    """Decoded text sequence from `self.tokens` above."""

    def append(self, token_ids: np.ndarray, decoded: str) -> None:
        """Appends to the generated tokens and decoded output."""
        assert len(token_ids.shape) == 1
        self.next_tokens = token_ids
        self.tokens.extend(token_ids)
        self.decoded += decoded

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
_ENGINE_LOCK = asyncio.Lock()


def _argmax_sampler(dtype: DType):
    logits_type = TensorType(dtype, ["batch", "vocab_size"])
    return Graph("argmax", ops.argmax, input_types=[logits_type])


class Llama3:
    """The overall interface to the Llama 3 model."""

    config: InferenceConfig
    _model: Model
    _sampler: Model
    _kv_manager: KVCacheManager
    _kv_params: KVCacheParams
    _tokenizer: Tokenizer

    def __init__(self, config: InferenceConfig):
        self.config = config

        assert config.weight_path is not None
        gguf_reader = gguf.GGUFReader(config.weight_path)

        self.params = _read_hyperparameters(config, gguf_reader)

        # Work around for older Llama 1/2 GGUFs, where the vocab size may be -1.
        # See https://github.com/ggerganov/llama.cpp/pull/4258.
        if self.params.vocab_size < 0:
            self.params.vocab_size = self._tokenizer.vocab_size

        dtype = (
            DType.float32 if self.params.quantization_encoding
            is not None else self.params.dtype
        )
        self._kv_params = KVCacheParams(
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            dtype=dtype,
            cache_strategy=config.cache_strategy,
        )

        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=config.max_cache_batch_size,
            max_seq_len=config.max_length,
            num_layers=self.params.n_layers,
            device=config.device,
        )

        session = InferenceSession(device=config.device)

        self._tokenizer = tokenizer_from_gguf(gguf_reader)
        self._model = self._load_model(
            session, config, self.params, gguf_reader
        )
        # logits are always float32 for now
        self._sampler = session.load(_argmax_sampler(DType.float32))

        if export_path := config.save_to_serialized_model_path:
            print(f"Exporting serialized model to {export_path}...")
            self._model._export_mef(export_path)

        self._n_heads = self.params.n_heads

    def _llama_graph_opaque(
        self,
        weights: GGUFWeights,
    ) -> Graph:
        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )
        valid_lengths_type = TensorType(DType.uint32, shape=["batch_size"])

        kv_cache_args = self._kv_manager.input_symbols()

        with Graph(
            "llama3",
            input_types=[
                tokens_type,
                attn_mask_type,
                valid_lengths_type,
                *kv_cache_args,
            ],
        ) as graph:
            model = transformer(graph, self.params, weights, self._kv_params)
            tokens, attention_mask, valid_lengths, *kv_cache = graph.inputs
            logits = model(
                tokens,
                attention_mask.cast(self.params.mask_dtype),
                valid_lengths,
                kv_cache,
            )
            graph.output(logits)
            return graph

    def _llama_graph(
        self,
        weights: GGUFWeights,
    ) -> Graph:
        if self.params.use_opaque:
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
                "batch_size",
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
            model = transformer(graph, self.params, weights, self._kv_params)
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

    async def _encode(self, prompt: str) -> list[int]:
        # Encodes a prompt using the tokenizer, raising a ValueError if the
        # prompt exceeds the configured maximum length.

        # Don't run compute-bound work on the main thread
        # however, it's not thread-safe, so make sure only one can
        # run at a time.
        async with _TOKENIZER_LOCK:
            encoded_prompt = await run_with_default_executor(
                self._tokenizer.encode, prompt
            )
        if len(encoded_prompt) >= self.config.max_length:
            msg = (
                f"Prompt length of {len(encoded_prompt)} is greater or equal to"
                " configured max model context length of"
                f" {self.config.max_length}."
            )
            raise ValueError(msg)

        return encoded_prompt

    async def _new_context_opaque(
        self, prompt: str, max_new_tokens: int | None = None
    ) -> Llama3Context:
        encoded_prompt = await self._encode(prompt)

        max_tokens_to_generate = _max_tokens_to_generate(
            len(encoded_prompt), self.config, max_new_tokens
        )
        seq_id = await self._kv_manager.claim(n=1)
        context = Llama3Context(
            prompt=prompt,
            max_tokens=len(encoded_prompt) + max_tokens_to_generate,
            cache_seq_id=seq_id[0],
        )

        context.append(np.array(encoded_prompt), prompt)
        return context

    async def new_context(
        self, prompt: str, max_new_tokens: int | None = None
    ) -> Llama3Context:
        if self.params.use_opaque:
            return await self._new_context_opaque(prompt, max_new_tokens)

        encoded_prompt = await self._encode(prompt)

        max_tokens_to_generate = _max_tokens_to_generate(
            len(encoded_prompt), self.config, max_new_tokens
        )
        seq_id = await self._kv_manager.claim(n=1)
        context = Llama3Context(
            prompt=prompt,
            max_tokens=len(encoded_prompt) + max_tokens_to_generate,
            cache_seq_id=seq_id[0],
        )
        context.append(np.array(encoded_prompt), prompt)
        return context

    async def next_token(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> dict[str, str | None]:
        res = {}

        # Don't run compute-bound work on the main thread
        # however, it's not thread-safe, so make sure only one can
        # run at a time.
        async with _ENGINE_LOCK:
            logits = await run_with_default_executor(
                self._execute, req_to_context_dict
            )
            (tokens,) = await run_with_default_executor(self._sampler, logits)
            tokens = tokens.to(CPU())

        next_tokens = dict(zip(req_to_context_dict, tokens.to_numpy()))
        for request_id, context in req_to_context_dict.items():
            next_token = next_tokens[request_id].astype(np.int64)
            decoded_token = self._tokenizer.decode(next_token)

            # Update context
            context.append(next_token.reshape(-1), decoded_token)

            # Mark completed requests by not including them in the response.
            if not context.is_done(self._tokenizer.eos_token_id):
                res[request_id] = decoded_token
            # TODO: MSDK-1084 Re-enable Cache release
            # Previously, we were automatically releasing completed sequences
            # back to the available cache pool, when the sequence completed
            # during the `next_token` loop. However, with the Contiguous
            # KV Cache, we cannot change the batch as sequences in the batch
            # are completed. This resulted in an indexing issue, trying to
            # retrieve details for a cache which was believed to be free by
            # the cache manager. We are temporarily stepping around this
            # auto-release, allowing the pipeline client in serving/benchmarking
            # to run completed prompts to preserve the batch. This should be
            # clarified at the API layer, identifying correct behaviour running
            # a completed sequence, and re-enabled if necessary.

        return res

    async def release(self, context: Llama3Context):
        await self._kv_manager.release(context.cache_seq_id)

    async def reset_cache(self):
        await self._kv_manager.reset_cache()

    def _execute_opaque(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> Tensor:
        context_batch = req_to_context_dict.values()
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get valid lengths: unpadded lengths of each token vector in the batch.
        batch_size = len(context_batch)
        unpadded_lengths = [ctx.seq_len for ctx in context_batch]
        valid_lengths = Tensor((batch_size,), DType.uint32, CPU())
        for n, valid_length in enumerate(unpadded_lengths):
            valid_lengths[n] = valid_length

        # Pad tokens and compute attention mask for the batch.
        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=[
                self._kv_manager.cache_lengths[seq_id]
                for seq_id in cache_seq_ids
            ],
            tokens=tokens,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        # Grab kv_collection.
        kv_cache_tensors = self._kv_manager.fetch(cache_seq_ids)

        # Execute model.
        logits = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self.config.device),
            Tensor.from_numpy(attn_mask).to(self.config.device),
            valid_lengths.to(self.config.device),
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
        if self.params.use_opaque:
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
            Tensor.from_numpy(next_tokens_batch).to(self.config.device),
            Tensor.from_numpy(attn_mask).to(self.config.device),
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


def _max_tokens_to_generate(
    prompt_size: int,
    config: InferenceConfig,
    max_new_tokens: Optional[int] = None,
) -> int:
    """Returns the max number of tokens to generate (including the prompt)."""
    max_new_tokens = (
        max_new_tokens if max_new_tokens is not None else config.max_new_tokens
    )
    if max_new_tokens < 0:
        return config.max_length - prompt_size
    return min(max_new_tokens, config.max_length - prompt_size)


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

    # If a quantization encoding is provided, the only caching strategy
    # available is the naive path. As such, overwrite this value to be
    # naive.
    if config.quantization_encoding.quantization_encoding:
        cache_strategy = KVCacheStrategy.NAIVE
    else:
        cache_strategy = config.cache_strategy

    return Hyperparameters(
        dtype=config.quantization_encoding.dtype,
        quantization_encoding=config.quantization_encoding.quantization_encoding,
        feed_forward_length=feed_forward_length,
        seq_len=seq_len,
        cache_strategy=cache_strategy,
        has_dedicated_output_weights=has_dedicated_output_weights,
        **configured_params,
    )
