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

from typing import TYPE_CHECKING

import gguf
import numpy as np
from max.driver import CPU, CUDA, Tensor, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, Graph, TensorType
from max.graph.weights import GGUFWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheStrategy,
    KVCacheParams,
    load_kv_manager,
)
from dataprocessing import TextContext, batch_padded_tokens_and_mask

from utils import gguf_utils
from .gguf import transformer
from .model.hyperparameters import Hyperparameters


def load_llama3_and_kv_manager(
    config: PipelineConfig,
    session: InferenceSession,
    vocab_size: int | None = None,
) -> tuple[Llama3, KVCacheManager]:
    reader = gguf.GGUFReader(config.weight_path)
    hyper_params = _read_hyperparameters(config, reader, vocab_size=vocab_size)
    kv_params = KVCacheParams(
        n_kv_heads=hyper_params.n_kv_heads,
        head_dim=hyper_params.head_dim,
        dtype=(
            DType.float32 if hyper_params.quantization_encoding
            is not None else hyper_params.dtype
        ),
        cache_strategy=config.cache_strategy,
    )

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=config.max_cache_batch_size,
        max_seq_len=hyper_params.seq_len,
        num_layers=hyper_params.n_layers,
        device=config.device,
        session=session,
    )
    model = Llama3(
        config,
        reader,
        hyper_params,
        kv_manager,
        session=session,
    )

    return model, kv_manager


class Llama3:
    """The Llama 3 model."""

    def __init__(
        self,
        config: PipelineConfig,
        reader: gguf.GGUFReader,
        hyperparams: Hyperparameters,
        kv_manager: KVCacheManager,
        *,
        session: InferenceSession | None = None,
    ):
        """Initializes the Llama3 model.

        Args:
            config: Model parameters.
            session: Optional InferenceSession to use to run the model.
            vocab_size: Vocabulary size of the model. Generally not required
              unless you're using an older model checkpoint.
        """
        self.config = config
        assert config.weight_path is not None
        self.reader = reader
        self.params = hyperparams
        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        self._kv_manager = kv_manager
        self._kv_params = self._kv_manager.params

        if session is None:
            session = InferenceSession(device=self._device)

        # Pre-allocate a buffer for input_row_offset in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offset_prealloc = Tensor.from_numpy(
            np.arange(config.max_cache_batch_size + 1, dtype=np.uint32)
        ).to(self._device)

        self._model = self._load_model(session, config, self.reader)

    def export_mef(self, export_path):
        self._model._export_mef(export_path)

    def _llama_graph_opaque(self, weights: GGUFWeights) -> Graph:
        tokens_type = TensorType(DType.int64, shape=["total_seq_len"])
        # NOTE: input_row_offset_len should be batch_size + 1.
        input_row_offset_type = TensorType(
            DType.uint32, shape=["input_row_offset_len"]
        )

        kv_cache_args = self._kv_manager.input_symbols()

        with Graph(
            "llama3",
            input_types=[tokens_type, input_row_offset_type, *kv_cache_args],
        ) as graph:
            model = transformer(
                graph,
                self.config.cache_strategy,
                self.params,
                weights,
                self._kv_params,
            )
            tokens, input_row_offset, *kv_cache = graph.inputs
            logits = model(tokens, kv_cache, input_row_offset=input_row_offset)
            graph.output(logits)
            return graph

    def _llama_graph(self, weights: GGUFWeights) -> Graph:
        if self.config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._llama_graph_opaque(weights)

        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )

        kv_inputs = self._kv_manager.input_symbols()

        with Graph(
            "llama3",
            input_types=[
                tokens_type,
                attn_mask_type,
                *kv_inputs,
            ],
        ) as graph:
            model = transformer(
                graph,
                self.config.cache_strategy,
                self.params,
                weights,
                self._kv_params,
            )
            tokens, attention_mask, k_cache, v_cache, start_pos, _ = (
                graph.inputs
            )
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
        config: PipelineConfig,
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

    def _prepare_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
        """Prepare the inputs for the first pass in multistep execution."""
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        if self.config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            # Get input_row_offset: start and end position of each batch in the
            # combined total_seq_len dimension.
            input_row_offset = Tensor.from_numpy(
                np.cumsum(
                    [0] + [ctx.seq_len for ctx in context_batch],
                    dtype=np.uint32,
                )
            ).to(self._device)

            # Create a ragged token vector of length: sum(len(t) for t in tokens).
            next_tokens_batch = np.concatenate(tokens)
            next_tokens_batch = Tensor.from_numpy(next_tokens_batch).to(
                self._device
            )

            return (next_tokens_batch, input_row_offset)
        else:
            # Pad tokens and compute attention mask for the batch.
            max_seq_len = self._kv_manager.max_sequence_length
            start_pos = [max_seq_len] * len(context_batch)
            next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
                start_pos=start_pos,
                tokens=tokens,
                pad_to_multiple_of=self.config.pad_to_multiple_of,
            )

            return (next_tokens_batch, attn_mask)

    def _prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        if self.config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            _, old_row_offsets = prev_model_inputs
            row_offsets_size = old_row_offsets.shape[0]
            next_row_offsets = self._input_row_offset_prealloc[
                :row_offsets_size
            ]
            next_token_inputs = (next_tokens, next_row_offsets)

            return next_token_inputs
        else:
            prev_tokens, prev_attn_mask = prev_model_inputs
            batch_size = prev_tokens.shape[0]
            start_pos = [prev_attn_mask.shape[-1]] * batch_size
            next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
                start_pos=start_pos,
                tokens=next_tokens,
                pad_to_multiple_of=self.config.pad_to_multiple_of,
            )
            next_token_inputs = (next_tokens_batch, attn_mask)

            return next_token_inputs

    def _execute(self, *model_inputs: Tensor) -> Tensor:
        """Executes the model and returns the raw results."""

        # Execute model.
        copy_inputs_to_device = (
            self.config.cache_strategy == KVCacheStrategy.NAIVE
        )
        return self._model.execute(
            *model_inputs, copy_inputs_to_device=copy_inputs_to_device
        )[0]


def _read_hyperparameters(
    config: PipelineConfig,
    reader: gguf.GGUFReader,
    *,
    vocab_size: int | None = None,
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

    # While Llama3.1 supports a context window of up to 128,000. The default is set
    # to 8000. The memory reserved within the KV cache is directly dependent on this value,
    # resulting in OOM memory on smaller machines, when set larger.
    seq_len = 8000
    if config.max_length > seq_len:
        print(
            "Warning: `max_length` is more than the supported context size"
            f"`max_length` is now set to {seq_len}"
        )
        config.max_length = seq_len
    else:
        seq_len = config.max_length

    # Newer llama models (>=3.2) may not use an output weight, and instead
    # re-use the embedding weight to compute the output logits.
    has_dedicated_output_weights = any(
        tensor.name == "output.weight" for tensor in reader.tensors
    )

    # Workaround for older Llama 1/2 GGUFs, where the vocab size may be -1.
    # See https://github.com/ggerganov/llama.cpp/pull/4258.
    if (configured_vocab_size := configured_params["vocab_size"]) < 0:
        if not vocab_size:
            raise ValueError(
                "Parsing a possible outdated GGUF where the vocab size is set"
                f" to {configured_vocab_size}. Please use a newer GGUF."
            )
        configured_params["vocab_size"] = vocab_size

    return Hyperparameters(
        dtype=config.quantization_encoding.dtype,
        quantization_encoding=config.quantization_encoding.quantization_encoding,
        feed_forward_length=feed_forward_length,
        seq_len=seq_len,
        has_dedicated_output_weights=has_dedicated_output_weights,
        **configured_params,
    )
