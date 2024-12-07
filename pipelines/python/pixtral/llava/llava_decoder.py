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

from dataclasses import dataclass
from typing import Union

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)
from nn import Embedding, Linear, LPLayerNorm, RMSNorm, TransformerBlock
from nn.layer import Layer


@dataclass
class Transformer(Layer):
    """Transformer model consisting for TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
    norm: Union[RMSNorm, LPLayerNorm]
    output: Linear
    embedding: Embedding  # this embedding layer is not used by __call__ but used by LlavaConditionalGeneration
    kv_params: KVCacheParams
    kv_collection_constructor: FetchContinuousBatchingKVCacheCollection

    def __call__(
        self,
        embeds,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> TensorValue:
        """Takes as input:
        embeds: embeddings of the sequence of text tokens and possibly images.
        shape = [batch_size, n_patches, hidden_dim]
        """
        h = embeds

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        # Predict with the last non-pad token (right-padded).
        if "input_row_offsets" in kwargs:
            # For ragged tensors gather the last tokens from packed dim 0.
            input_row_offsets: TensorValueLike = kwargs["input_row_offsets"]
            last_token_indices = input_row_offsets[1:] - 1  # type: ignore
            # Should be: last_token = h[last_token_indices]
            last_token = ops.gather(h, last_token_indices, axis=0)
        else:
            # For padded tensors, use `gather_nd`.
            # Unsqueeze since `gather_nd` expects a static last dim.
            valid_lengths: TensorValueLike = kwargs["valid_lengths"]
            last_token = ops.gather_nd(
                h,
                indices=ops.unsqueeze(valid_lengths - 1, -1),  # type: ignore
                batch_dims=1,
            )

        # Always return float32 logits, no matter the activation type
        return ops.cast(self.output(self.norm(last_token)), DType.float32)
