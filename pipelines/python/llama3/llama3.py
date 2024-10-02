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

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import gguf
import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import GGUFWeights
from tokenizers import Tokenizer

from nn.kv_cache import (
    ContiguousKVCacheCollectionType,
    ContiguousKVCacheManager,
    KVCacheParams,
    NaiveKVCache,
)
from utils import gguf_utils, tokenizer_from_gguf

from .causal_attention_mask import causal_attention_mask
from .collate_batch import collate_batch
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

    cache_seq_id: int | None = None
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


def _llama_graph_opaque(
    params: Hyperparameters,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
) -> Graph:
    tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
    attn_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
    )
    valid_lengths_type = TensorType(DType.uint32, shape=["batch_size"])
    cache_type = ContiguousKVCacheCollectionType()

    with Graph(
        "llama3",
        input_types=[
            tokens_type,
            attn_mask_type,
            valid_lengths_type,
            cache_type,
        ],
    ) as graph:
        model = transformer(graph, params, weights, kv_params)
        tokens, attention_mask, *kv_cache = graph.inputs
        logits = model(tokens, attention_mask.cast(params.dtype), *kv_cache)
        graph.output(logits[:, -1])
        return graph


def _llama_graph(
    params: Hyperparameters,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
) -> Graph:
    if params.use_opaque:
        return _llama_graph_opaque(params, weights, kv_params)

    tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
    attn_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
    )

    cache_type = TensorType(
        DType.float32,
        shape=[
            "start_pos",
            params.n_layers,
            "batch_size",
            params.n_kv_heads,
            params.head_dim,
        ],
    )

    with Graph(
        "llama3",
        input_types=[tokens_type, attn_mask_type, cache_type, cache_type],
    ) as graph:
        model = transformer(graph, params, weights, kv_params)
        tokens, attention_mask, *kv_cache = graph.inputs
        logits, k_update, v_update = model(
            tokens, attention_mask.cast(params.dtype), *kv_cache
        )
        graph.output(logits[:, -1], k_update, v_update)
        return graph


class Llama3:
    """The overall interface to the Llama 3 model."""

    config: InferenceConfig
    _model: Model
    _kv_cache: NaiveKVCache
    _kv_manager: ContiguousKVCacheManager
    _sessions: set[str]
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

        self._sessions = set[str]()

        dtype = (
            DType.float32 if self.params.quantization_encoding
            is not None else self.params.dtype
        )
        self._kv_params = KVCacheParams(
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            dtype=dtype,
            device=config.device,
        )

        session = InferenceSession(device=config.device)

        self._tokenizer = tokenizer_from_gguf(gguf_reader)
        self._model = self._load_model(
            session, config, self.params, gguf_reader
        )

        if export_path := config.save_to_serialized_model_path:
            print(f"Exporting serialized model to {export_path}...")
            self._model._export_mef(export_path)

        if self.params.use_opaque:
            self._kv_manager = ContiguousKVCacheManager(
                params=self._kv_params,
                max_batch_size=config.batch_size,
                max_seq_len=config.max_length,
                num_layers=self.params.n_layers,
                session=session,
                device=config.device,
            )
        else:
            self._kv_cache = NaiveKVCache(
                self.params.seq_len,
                self.config.batch_size,
                self.params.n_layers,
                self.params.n_kv_heads,
                self.params.head_dim,
            )

        self._n_heads = self.params.n_heads

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
            graph = _llama_graph(params, self._weights, self._kv_params)
            print("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights
            )

    def _encode(self, prompt: str) -> list[int]:
        # Encodes a prompt using the tokenizer, raising a ValueError if the
        # prompt exceeds the configured maximum length.
        encoded_prompt = self._tokenizer.encode(prompt)
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
        encoded_prompt = self._encode(prompt)

        max_tokens_to_generate = _max_tokens_to_generate(
            len(encoded_prompt), self.config, max_new_tokens
        )
        seq_id = await self._kv_manager.claim(batch_size=1)
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

        encoded_prompt = self._encode(prompt)

        max_tokens_to_generate = _max_tokens_to_generate(
            len(encoded_prompt), self.config, max_new_tokens
        )
        context = Llama3Context(
            prompt=prompt,
            max_tokens=len(encoded_prompt) + max_tokens_to_generate,
        )
        context.append(np.array(encoded_prompt), prompt)
        return context

    async def next_token(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> dict[str, str | None]:
        # TODO(MSDK-889) - Consider moving request/cache mgmt out of next_token.

        res = {}
        if self._sessions ^ req_to_context_dict.keys():
            self._sessions = set(req_to_context_dict.keys())
            # TODO: MSDK-1020 We should not reset the cache here unilaterally
            # as the cache here uses an independent method for _sessions
            # management this should be fixed.
            if not self.params.use_opaque:
                await self.reset_cache()

        req_id_to_logits_dict = self._execute(req_to_context_dict)

        for request_id, context in req_to_context_dict.items():
            # TODO: Add a weighted sampler here.
            # Get argmax of the logits of the last token.
            next_token = req_id_to_logits_dict[request_id].argmax(axis=-1)[-1]
            decoded_token = self._tokenizer.decode(next_token)

            # Update context
            context.append(next_token.reshape(-1), decoded_token)
            # Add back to dictionary

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
        if self.params.use_opaque:
            await self._kv_manager.release(context.cache_seq_id)  # type: ignore

    async def reset_cache(self):
        if self.params.use_opaque:
            await self._kv_manager.reset_cache()
        else:
            self._kv_cache.sequence_length = 0

    def _execute_opaque(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> dict[str, Tensor]:
        context_batch = req_to_context_dict.values()
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get valid lengths: unpadded lengths of each token vector in the batch.
        batch_size = len(context_batch)
        unpadded_lengths = [ctx.seq_len for ctx in context_batch]
        valid_lengths = Tensor((batch_size,), DType.uint32, CPU())
        for n, valid_length in enumerate(unpadded_lengths):
            valid_lengths[n] = valid_length

        # Grab attention mask.
        attn_mask = causal_attention_mask(
            original_start_pos=list(self._kv_manager.cache_lengths.values()),
            original_seq_len=[ctx.seq_len for ctx in context_batch],
        ).astype(np.float32)

        # Grab kv_collection.
        kv_collection = self._kv_manager.fetch(
            [ctx.cache_seq_id for ctx in context_batch]
        )

        # Create batched input token tensor by padding all input token tensors
        # to the maximum sequence length in the batch.
        next_tokens_batch = collate_batch(
            tokens, batch_size=self.config.batch_size
        )

        # Execute model.
        batch_logits = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self.config.device),
            Tensor.from_numpy(attn_mask).to(self.config.device),
            valid_lengths,
            kv_collection,
        )[0]

        # Copy logits from device to host.
        batch_logits = np.from_dlpack(batch_logits.to(CPU()))

        self._kv_manager.step(
            valid_lengths={
                ctx.cache_seq_id: ctx.seq_len for ctx in context_batch
            }
        )

        return dict(zip(req_to_context_dict, batch_logits))

    def _execute(
        self, req_to_context_dict: dict[str, Llama3Context]
    ) -> dict[str, np.ndarray]:
        """Executes the model and returns the raw results."""
        if self.params.use_opaque:
            return self._execute_opaque(req_to_context_dict)

        context_batch = req_to_context_dict.values()
        tokens = [ctx.next_tokens for ctx in context_batch]
        batch_size = len(context_batch)

        # Create batched input token tensor by padding all input token tensors
        # to the maximum sequence length in the batch.
        next_tokens_batch = collate_batch(
            tokens, batch_size=self.config.batch_size
        )

        # Grab attention mask.
        attn_mask = causal_attention_mask(
            original_start_pos=(
                [self._kv_cache.sequence_length] * len(req_to_context_dict)
            ),
            original_seq_len=[ctx.seq_len for ctx in context_batch],
        ).astype(np.float32)

        # Execute model.
        logits, k_cache, v_cache = self._model.execute(
            Tensor.from_numpy(next_tokens_batch).to(self.config.device),
            Tensor.from_numpy(attn_mask).to(self.config.device),
            Tensor.from_numpy(self._kv_cache.keys_view(batch_size)).to(
                self.config.device
            ),
            Tensor.from_numpy(self._kv_cache.values_view(batch_size)).to(
                self.config.device
            ),
        )

        logits = np.from_dlpack(logits.to(CPU()))
        k_cache = np.from_dlpack(k_cache.to(CPU()))
        v_cache = np.from_dlpack(v_cache.to(CPU()))

        self._kv_cache.update(k_cache, v_cache)

        logits_to_return = {}

        # Since req_to_context_dict dict is ordered as it was passed in from the
        # input, we just iterate over the req_ids in that order and assign
        # logits that way.
        for curr_index, req_id in enumerate(req_to_context_dict):
            logits_to_return[req_id] = logits[curr_index]

        return logits_to_return


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

    return Hyperparameters(
        dtype=config.quantization_encoding.dtype,
        quantization_encoding=config.quantization_encoding.quantization_encoding,
        feed_forward_length=feed_forward_length,
        seq_len=seq_len,
        force_naive_kv_cache=config.force_naive_kv_cache,
        has_dedicated_output_weights=has_dedicated_output_weights,
        **configured_params,
    )
