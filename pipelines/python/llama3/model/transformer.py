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
from typing import Tuple

from max.graph import ValueLike, TensorValue, ops
from typing import List

from .attention import Attention
from .mlp import MLP, Linear
from .norm import RMSNorm
from .embedding import Embedding


@dataclass
class TransformerBlock:
    """
    Stacks Attention, FeedForward, and RMSNorm layers
    into single transformer block.
    """

    attention: Attention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self,
        x: ValueLike,
        attention_mask: ValueLike,
        k_cache: ValueLike,
        v_cache: ValueLike,
    ) -> Tuple[TensorValue, TensorValue, TensorValue]:
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(x), attention_mask, k_cache, v_cache
        )

        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))

        return h, k_cache_update, v_cache_update


@dataclass
class Transformer:
    """
    Transformer model consisting of TransformerBlock layers.
    """

    dim: int
    n_heads: int
    layers: List[TransformerBlock]
    norm: RMSNorm
    output: Linear
    theta: float
    embedding: Embedding

    def __call__(self, tokens, attention_mask, k_cache, v_cache):
        h = self.embedding(tokens)
        k_cache_updates = []
        v_cache_updates = []
        for i in range(len(self.layers)):
            h, k_cache_layer_update, v_cache_layer_update = self.layers[i](
                h,
                attention_mask,
                k_cache[:, i],
                v_cache[:, i],
            )
            k_cache_updates.append(ops.transpose(k_cache_layer_update, 0, 1))
            v_cache_updates.append(ops.transpose(v_cache_layer_update, 0, 1))

        return (
            self.output(self.norm(h)),
            ops.stack(k_cache_updates, axis=1),
            ops.stack(v_cache_updates, axis=1),
        )
