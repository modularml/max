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

"""Llama 3.2 Transformer Vision Language Model cross attention decoder."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.graph.weights import SafetensorWeights
from nn import MLP, Linear, RMSNorm
from nn.layer import Layer

from .cache import Cache
from .hyperparameters import TextHyperparameters


# TODO(MAXCORE-145): Implement this!
@dataclass
class SelfAttentionDecoderLayer(Layer):
    params: TextHyperparameters

    # hidden_states: shape=[1, 1, 4096], dtype=torch.bfloat16
    # cross_attention_states: value=None
    # cross_attention_mask: shape=[1, 1, 1, 4100], dtype=torch.bfloat16
    # attention_mask: value=None
    # full_text_row_masked_out_mask: shape=[1, 1, 1, 1], dtype=torch.bfloat16
    # position_ids: shape=[1, 1], dtype=torch.int64
    # past_key_value: value=DynamicCache()
    # output_attentions: value=False
    # use_cache: value=True
    # cache_position: shape=[1], dtype=torch.int64
    # position_embeddings[0]: shape=[1, 1, 128], dtype=torch.bfloat16
    # position_embeddings[1]: shape=[1, 1, 128], dtype=torch.bfloat16
    #   Output Shapes: [[1, 1, 4096]]
    #   Output DTypes: [torch.bfloat16]
    def __call__(
        self,
        hidden_states: TensorValue,
        cross_attention_states: TensorValue | None = None,
        cross_attention_mask: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        full_text_row_masked_out_mask: tuple[TensorValue, TensorValue]
        | None = None,
        position_ids: TensorValue | None = None,
        past_key_value: Cache | None = None,
        cache_position: TensorValue | None = None,
        position_embeddings: TensorValue | None = None,
    ) -> tuple[TensorValue, Cache | None]:
        return (hidden_states, past_key_value)
