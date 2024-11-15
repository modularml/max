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
import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import GGUFWeights
from max.pipelines import (
    PipelineModel,
    TextContext,
    SupportedEncoding,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    load_kv_manager,
    KVCacheStrategy,
)

from dataprocessing import batch_padded_tokens_and_mask
from .gguf import transformer


class Llama3Model(PipelineModel):
    def execute(self, *model_inputs: Tensor) -> Tensor:
        return self.model.execute(*model_inputs, copy_inputs_to_device=False)[0]

    def _prepare_continuous_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Get input_row_offset: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offset = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        next_tokens_batch = np.concatenate(tokens)
        next_tokens_batch = Tensor.from_numpy(next_tokens_batch).to(
            self._device
        )

        return (next_tokens_batch, input_row_offset)

    def _prepare_naive_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self.kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(context_batch)
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        return (next_tokens_batch, attn_mask)

    def prepare_initial_token_inputs(
        self, context_batch: list[TextContext]
    ) -> tuple[Tensor, ...]:
        """Prepare the inputs for the first pass in multistep execution."""
        if self.pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._prepare_continuous_initial_token_inputs(context_batch)
        else:
            return self._prepare_naive_initial_token_inputs(context_batch)

    def _prepare_continuous_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ):
        _, old_row_offsets = prev_model_inputs
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offset_prealloc[:row_offsets_size]
        next_token_inputs = (next_tokens, next_row_offsets)

        return next_token_inputs

    def _prepare_naive_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ):
        prev_tokens, prev_attn_mask = prev_model_inputs
        batch_size = prev_tokens.shape[0]
        start_pos = [prev_attn_mask.shape[-1]] * batch_size
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=next_tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )
        next_token_inputs = (next_tokens_batch, attn_mask)

        return next_token_inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        if self.pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._prepare_continuous_next_token_inputs(
                next_tokens, prev_model_inputs
            )
        else:
            return self._prepare_naive_next_token_inputs(
                next_tokens, prev_model_inputs
            )

    def _get_kv_params(self) -> KVCacheParams:
        cache_dtype = (
            DType.float32 if self.pipeline_config.quantization_encoding.quantization_encoding
            is not None else self.pipeline_config.dtype
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.hidden_size
            // self.pipeline_config.huggingface_config.num_attention_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(self, session: InferenceSession) -> KVCacheManager:
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            device=self.pipeline_config.device,
            session=session,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # Pre-allocate a buffer for input_row_offset in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offset_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.device)

        # Read in weights.
        self._weights = self.pipeline_config.load_weights()

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():
                weights_registry[name] = tensor.data

            logging.info("Loading serialized model from ", serialized_path)

            return session.load(
                serialized_path, weights_registry=weights_registry
            )

        else:
            logging.info("Building model...")
            graph = self._build_graph(self._weights)
            logging.info("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights
            )

    def _build_opaque_graph(self, weights: GGUFWeights) -> Graph:
        tokens_type = TensorType(DType.int64, shape=["total_seq_len"])
        # NOTE: input_row_offset_len should be batch_size + 1.
        input_row_offset_type = TensorType(
            DType.uint32, shape=["input_row_offset_len"]
        )

        kv_cache_args = self.kv_manager.input_symbols()

        with Graph(
            "llama3",
            input_types=[tokens_type, input_row_offset_type, *kv_cache_args],
        ) as graph:
            model = transformer(
                graph,
                self.pipeline_config,
                weights,
                self._get_kv_params(),
            )
            tokens, input_row_offset, *kv_cache = graph.inputs
            logits = model(tokens, kv_cache, input_row_offset=input_row_offset)
            graph.output(logits)
            return graph

    def _build_graph(self, weights: GGUFWeights) -> Graph:
        if self.pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
            return self._build_opaque_graph(weights)

        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )

        kv_inputs = self.kv_manager.input_symbols()

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
                self.pipeline_config,
                weights,
                self._get_kv_params(),
            )
            tokens, attention_mask, k_cache, v_cache, start_pos, _ = (
                graph.inputs
            )
            mask_dtype = (
                self.pipeline_config.dtype if self.pipeline_config.quantization_encoding
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
