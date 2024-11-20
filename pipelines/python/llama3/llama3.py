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

import logging

import gguf
import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import GGUFWeights
from max.pipelines import PipelineConfig, SupportedEncoding, TextContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheStrategy,
    KVCacheParams,
    load_kv_manager,
)
from dataprocessing import batch_padded_tokens_and_mask

from .gguf import transformer

logger = logging.getLogger(__name__)


def load_llama3_and_kv_manager(
    config: PipelineConfig,
    session: InferenceSession,
) -> tuple[Llama3, KVCacheManager]:
    reader = gguf.GGUFReader(config.weight_path)
    cache_dtype = (
        DType.float32 if config.quantization_encoding.quantization_encoding
        is not None else config.dtype
    )
    kv_params = KVCacheParams(
        n_kv_heads=config.huggingface_config.num_key_value_heads,
        head_dim=config.huggingface_config.hidden_size
        // config.huggingface_config.num_attention_heads,
        dtype=cache_dtype,
        cache_strategy=config.cache_strategy,
    )

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=config.max_cache_batch_size,
        max_seq_len=config.huggingface_config.max_seq_len,
        num_layers=config.huggingface_config.num_hidden_layers,
        devices=[config.device],
        session=session,
    )
    model = Llama3(
        config,
        reader,
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
        kv_manager: KVCacheManager,
        *,
        session: InferenceSession | None = None,
    ) -> None:
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
        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        self._kv_manager = kv_manager
        self._kv_params = self._kv_manager.params

        if session is None:
            session = InferenceSession(devices=[self._device])

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
                self.config,
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
                self.config,
                weights,
                self._kv_params,
            )
            tokens, attention_mask, k_cache, v_cache, start_pos, _ = (
                graph.inputs
            )
            mask_dtype = (
                self.config.dtype if self.config.quantization_encoding
                in [
                    SupportedEncoding.float32,
                    SupportedEncoding.bfloat16,
                ] else DType.float32
            )
            logits, end_pos = model(
                tokens,
                attention_mask.cast(mask_dtype),
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
            logging.info(
                "Loading serialized model from %s ...", serialized_path
            )
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            logging.info("Building model...")
            graph = self._llama_graph(self._weights)
            logging.info("Compiling...")
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
