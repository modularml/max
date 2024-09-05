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

import os
from dataclasses import dataclass
from typing import Tuple

import gguf
import numpy as np
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import GGUFWeights

from utils import gguf_utils, tokenizer_from_gguf

from .config import InferenceConfig, SupportedEncodings
from .gguf import transformer
from .kernel_names import KVCacheKernelNames
from .kv_cache import KVCache
from .model.hyperparameters import Hyperparameters


@dataclass
class Llama3Context:
    """The context for text generation using a Llama 3 model."""

    prompt: str
    prompt_size: int
    max_tokens: int
    next_token: np.ndarray
    next_decoded: str
    output_token_count: int = 0  # Number of generated tokens so far.
    output_sequence: str = ""  # Generated text sequence from tokens so far.

    def is_done(self, eos: str) -> bool:
        if self.next_decoded == eos:
            return True
        max_output_tokens = max(1, self.max_tokens - self.prompt_size)
        if self.output_token_count == max_output_tokens:
            return True
        return False


def _llama_graph(
    batch_size: int,
    params: Hyperparameters,
    weights: GGUFWeights,
    kernel_names: KVCacheKernelNames,
) -> Graph:
    tokens_type = TensorType(DType.int64, shape=[batch_size, "seq_len"])
    attn_mask_type = TensorType(DType.bool, shape=[batch_size, "post_seq_len"])
    cache_type = TensorType(
        DType.float32,
        shape=[
            "start_pos",
            params.n_layers,
            batch_size,
            params.n_kv_heads,
            params.head_dim,
        ],
    )

    with Graph(
        "llama3",
        input_types=[tokens_type, attn_mask_type, cache_type, cache_type],
    ) as graph:
        model = transformer(graph, params, weights, kernel_names)
        logits, k_update, v_update = model(*graph.inputs)
        graph.output(logits[:, -1], k_update, v_update)
        return graph


class Llama3:
    """The overall interface to the Llama 3 model."""

    config: InferenceConfig
    _model: Model
    _kv_cache: KVCache
    _sessions: set[str]
    _kernel_names: KVCacheKernelNames

    def __init__(self, config: InferenceConfig):
        self.config = config

        assert config.weight_path is not None
        gguf_reader = gguf.GGUFReader(config.weight_path)

        params = _read_hyperparameters(
            config.quantization_encoding, gguf_reader
        )

        # Work around for older Llama 1/2 GGUFs, where the vocab size may be -1.
        # See https://github.com/ggerganov/llama.cpp/pull/4258.
        if params.vocab_size < 0:
            params.vocab_size = self._tokenizer.vocab_size

        self._kv_cache = KVCache(
            params.seq_len,
            config.batch_size,
            params.n_layers,
            params.n_kv_heads,
            params.head_dim,
        )
        self._sessions = set[str]()

        dtype = (
            DType.float32 if params.quantization_encoding
            is not None else params.dtype
        )
        self._kernel_names = KVCacheKernelNames(
            n_kv_heads=params.n_kv_heads,
            head_dim=params.head_dim,
            dtype=dtype,
            device=config.device,
        )

        self._model = self._load_model(config, params, gguf_reader)
        self._tokenizer = tokenizer_from_gguf(gguf_reader)

    def _load_model(
        self,
        config: InferenceConfig,
        params: Hyperparameters,
        reader: gguf.GGUFReader,
    ) -> Model:
        session = InferenceSession(device=config.device)
        if serialized_path := config.serialized_model_path:
            print("Loading serialized model from", serialized_path, "...")
            return session.load(serialized_path)
        else:
            # TODO(GRA-964): Revert #46659 when memory planning is done.
            self._weights = GGUFWeights(
                reader,
                use_resource="MODULAR_USE_EXTERNAL_WEIGHTS" not in os.environ,
            )
            print("Building model...")
            graph = _llama_graph(
                config.batch_size, params, self._weights, self._kernel_names
            )
            print("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights
            )

    def _get_attention_mask(self, n: int):
        mask = np.ones(shape=(1, n)).astype(bool)
        return mask

    async def new_context(self, prompt: str) -> Llama3Context:
        encoded_prompt = self._tokenizer.encode(prompt)
        prompt_size = len(encoded_prompt)
        return Llama3Context(
            prompt=prompt,
            prompt_size=prompt_size,
            max_tokens=_max_tokens_to_generate(prompt_size, self.config),
            next_token=np.array(encoded_prompt).reshape(1, -1),
            next_decoded="",
        )

    async def next_token(
        self, batch: dict[str, Llama3Context]
    ) -> dict[str, str | None]:
        # TODO(MSDK-889) - Consider moving request/cache mgmt out of next_token.
        # Note: assuming a single request.
        assert len(batch) == self.config.batch_size == 1
        request_id, context = next(iter(batch.items()))

        if context.is_done(self._tokenizer.eos_token):
            self._sessions.remove(request_id)
            return {request_id: None}

        if request_id not in self._sessions:
            self._sessions.add(request_id)
            self._reset_cache()

        logits, _, _ = self._execute(context)

        # TODO: Add a weighted sampler here.
        # Get argmax of the logits of the last token.
        next_token = logits.argmax(axis=-1)[-1]
        decoded_token = self._tokenizer.decode(next_token)

        # Update context
        context.next_token = next_token.reshape(1, -1)
        context.next_decoded = decoded_token
        context.output_sequence += decoded_token
        context.output_token_count += 1

        return {request_id: decoded_token}

    def _reset_cache(self):
        # This feels really contrived, but it's because our KV cache setup
        # just doesn't meaningfully support batch size > 1 yet.
        self._kv_cache.sequence_length = 0

    def _execute(self, context: Llama3Context) -> Tuple[np.ndarray, ...]:
        """Executes the model and returns the raw results."""
        cache = self._kv_cache
        output_names = [t.name for t in self._model.output_metadata]

        result = self._model.execute(
            input0=context.next_token,
            input1=self._get_attention_mask(
                cache.sequence_length + context.next_token.shape[1]
            ),
            input2=cache.keys_view(),
            input3=cache.values_view(),
        )

        logits, k_cache, v_cache = (result[o] for o in output_names)
        self._kv_cache.update(k_cache, v_cache)
        return logits, k_cache, v_cache


def _max_tokens_to_generate(prompt_size: int, config: InferenceConfig) -> int:
    """Returns the max number of tokens to generate (including the prompt)."""
    if config.max_new_tokens < 0:
        return config.max_length
    return min(config.max_new_tokens + prompt_size, config.max_length)


def _read_hyperparameters(
    model_encoding: SupportedEncodings, reader: gguf.GGUFReader
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
    return Hyperparameters(
        dtype=model_encoding.dtype,
        quantization_encoding=model_encoding.quantization_encoding,
        feed_forward_length=feed_forward_length,
        **configured_params,
    )
