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
from dataprocessing import batch_padded_tokens_and_mask, collate_batch
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, Graph, TensorType
from max.graph.weights import GGUFWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)

from utils import gguf_utils

from .gguf import transformer
from .model.hyperparameters import Hyperparameters

if TYPE_CHECKING:
    from nn.context import TextContext


class Llama3:
    """The Llama 3 model."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        session: InferenceSession | None = None,
        vocab_size: int | None = None,
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
        self.reader = gguf.GGUFReader(config.weight_path)
        self.params = _read_hyperparameters(
            self.config, self.reader, vocab_size=vocab_size
        )
        device_spec = self.config.device_spec
        self._device = CPU(
            device_spec.id
        ) if device_spec.device_type == "cpu" else CUDA(device_spec.id)

        self._kv_params = KVCacheParams(
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            dtype=(
                DType.float32 if self.params.quantization_encoding
                is not None else self.params.dtype
            ),
            cache_strategy=config.cache_strategy,
        )

        self._kv_manager = load_kv_manager(
            params=self._kv_params,
            max_cache_batch_size=config.max_cache_batch_size,
            max_seq_len=self.params.seq_len,
            num_layers=self.params.n_layers,
            device=self._device,
        )
        if session is None:
            session = InferenceSession(device=self._device)

        self._model = self._load_model(
            session, config, self.params, self.reader
        )

    def export_mef(self, export_path):
        self._model._export_mef(export_path)

    def _llama_graph_opaque(self, weights: GGUFWeights) -> Graph:
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
                graph,
                self.config.cache_strategy,
                self.params,
                weights,
                self._kv_params,
            )
            tokens, valid_lengths, *kv_cache = graph.inputs
            logits = model(
                tokens,
                valid_lengths,
                kv_cache,
            )
            graph.output(logits)
            return graph

    def _llama_graph(self, weights: GGUFWeights) -> Graph:
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
                graph,
                self.config.cache_strategy,
                self.params,
                weights,
                self._kv_params,
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
        config: PipelineConfig,
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

    def release(self, context: TextContext):
        self._kv_manager.release(context.cache_seq_id)

    def _execute_opaque(
        self, req_to_context_dict: dict[str, TextContext]
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

    def _execute(self, req_to_context_dict: dict[str, TextContext]) -> Tensor:
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
    # to 512. The memory reserved within the KV cache is directly dependent on this value,
    # resulting in OOM memory on smaller machines, when set larger.
    seq_len = 512
    if config.max_length is not None:
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
