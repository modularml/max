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

from typing import Any

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import PipelineConfig, TextContext
from max.pipelines.interfaces import TokenGenerator
from max.pipelines.sampling import token_sampler
from max.profiler import traced

from .llama3 import load_llama3_and_kv_manager


class Llama3TokenGenerator(TokenGenerator[TextContext]):
    """Token Generator for the Llama 3 model."""

    def __init__(self, config: PipelineConfig, eos: int):
        self.config = config
        self.eos = eos

        self._device = config.device
        session = InferenceSession(devices=[self._device])

        self.model, self._kv_manager = load_llama3_and_kv_manager(
            config,
            session,
        )

        # Logits are always float32 for now
        self._sampler = session.load(token_sampler(config.top_k, DType.float32))

        if export_path := config.save_to_serialized_model_path:
            print(f"Exporting serialized model to {export_path}...")
            self.model.export_mef(export_path)

    @traced
    def next_token(
        self, batch: dict[str, TextContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        # Flatten our batch for consistent indexing
        context_batch = list(batch.values())

        # Claim cache rows for our batch
        for context in context_batch:
            if context.cache_seq_id in self._kv_manager.slots_remaining:
                self._kv_manager.external_claim([context.cache_seq_id])

        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]

        # Prepare inputs for the first token in multistep execution
        model_inputs = self.model._prepare_initial_token_inputs(context_batch)
        kv_cache_inputs = self._kv_manager.fetch(cache_seq_ids)

        # Multistep execution loop
        batch_size = len(context_batch)
        generated_tokens = Tensor.from_numpy(
            np.zeros((batch_size, 0), dtype=np.int64)
        ).to(self._device)
        curr_step_inputs = model_inputs
        for i in range(num_steps):
            # Execute the model and get next tokens
            logits = self.model._execute(*curr_step_inputs, *kv_cache_inputs)
            new_tokens, generated_tokens = self._sampler(  # type: ignore
                logits, generated_tokens
            )[:2]

            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                break

            # Prepare inputs for the next token in multistep execution
            kv_cache_inputs = self._kv_manager.increment_cache_lengths(  # type: ignore
                kv_cache_inputs,  # type: ignore
                curr_step_inputs,
            )
            curr_step_inputs = self.model._prepare_next_token_inputs(
                new_tokens, curr_step_inputs  # type: ignore
            )

        # Actually update the cache lengths in our kv_cache manager
        self._kv_manager.step(
            valid_lengths={
                ctx.cache_seq_id: ctx.seq_len + num_steps - 1
                for ctx in context_batch
            }
        )

        # Do the copy to host for each token generated.
        generated_tokens = generated_tokens.to(CPU()).to_numpy()  # type: ignore

        # Prepare the response, pruning away completed requests as we go.
        res: list[dict[str, Any]] = []
        is_done = {r: False for r in batch.keys()}
        for i in range(num_steps):
            step_res = {}
            next_tokens = dict(zip(batch, generated_tokens[:, i]))  # type: ignore
            for request_id, context in batch.items():
                if is_done[request_id]:
                    continue

                next_token = next_tokens[request_id]

                # Update context
                context.update(new_tokens=next_token.reshape(-1))

                # Mark completed requests by not including them in the response.
                if not context.is_done(self.eos):
                    step_res[request_id] = next_token
                else:
                    is_done[request_id] = True

            res.append(step_res)

        return res

    def release(self, context: TextContext):
        self._kv_manager.release(context.cache_seq_id)
