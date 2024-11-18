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

import logging
import numpy as np
from dataprocessing import (
    causal_attention_mask_with_alibi,
    collate_batch,
)
from max.pipelines import PipelineModel, TextContext, PipelineConfig
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    load_kv_manager,
)
from max.driver import Tensor
from max.engine import InferenceSession, Model
from .graph import _build_graph


class ReplitModel(PipelineModel):
    def execute(self, *model_inputs: Tensor) -> Tensor:  # type: ignore
        return self.model.execute(*model_inputs, copy_inputs_to_device=False)[0]  # type: ignore

    def prepare_initial_token_inputs(
        self, context_batch: list[TextContext]  # type: ignore
    ) -> tuple[Tensor, ...]:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]
        unpadded_lengths = [ctx.seq_len for ctx in context_batch]

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self.kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(context_batch)
        next_tokens_batch, _ = collate_batch(
            tokens,
            batch_size=len(tokens),
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        attention_mask = causal_attention_mask_with_alibi(
            original_start_pos=start_pos,
            original_seq_len=[len(t) for t in tokens],
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
            alibi_bias_max=self.pipeline_config.huggingface_config.attn_config[
                "alibi_bias_max"
            ],
            n_heads=self.pipeline_config.huggingface_config.n_heads,
        )

        return (
            Tensor.from_numpy(next_tokens_batch).to(
                self.pipeline_config.device
            ),
            Tensor.from_numpy(attention_mask).to(self.pipeline_config.device),
            Tensor.from_numpy(np.array(unpadded_lengths, np.uint32)).to(
                self.pipeline_config.device
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        # Update valid_lengths by one for all inputs.
        prev_tokens, prev_attention_mask, valid_lengths = prev_model_inputs
        valid_lengths += 1  # type: ignore

        batch_size = prev_tokens.shape[0]
        start_pos = [prev_attention_mask.shape[-1]] * batch_size
        next_tokens_batch = collate_batch(
            next_tokens,  # type: ignore
            batch_size=batch_size,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        attention_mask = causal_attention_mask_with_alibi(
            original_start_pos=start_pos,
            original_seq_len=[len(t) for t in next_tokens],  # type: ignore
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
            alibi_bias_max=self.pipeline_config.huggingface_config.attn_config[
                "alibi_bias_max"
            ],
            n_heads=self.pipeline_config.huggingface_config.n_heads,
        )

        # I believe, next_tokens_batch & valid_lengths, should already be resident on the GPU.
        # The attention mask is a new tensor, and thus has to be moved over.
        return (
            Tensor.from_numpy(next_tokens_batch),  # type: ignore
            Tensor.from_numpy(attention_mask).to(self.pipeline_config.device),
            Tensor.from_numpy(valid_lengths),  # type: ignore
        )

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.attn_config[
                "kv_n_heads"
            ],
            head_dim=self.pipeline_config.huggingface_config.d_model
            // self.pipeline_config.huggingface_config.n_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(self, session: InferenceSession) -> KVCacheManager:
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.n_layers,
            device=self.pipeline_config.device,
            session=session,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # Read in weights.
        self._weights = self.pipeline_config.load_weights()

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():
                weights_registry[name] = tensor.data

            logging.info("Loading serialized model from ", serialized_path)

            return session.load(
                serialized_path, weights_registry=weights_registry  # type: ignore
            )

        else:
            logging.info("Building model...")
            graph = _build_graph(
                self.pipeline_config,
                self._weights,
                self._get_kv_params(),
                kv_manager=self.kv_manager,
            )
            logging.info("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights  # type: ignore
            )
