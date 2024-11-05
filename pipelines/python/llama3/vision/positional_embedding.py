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

"""Llama 3.2 Transformer Vision Model Positional Embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops
from nn import Embedding
from nn.layer import Layer

from .hyperparameters import VisionHyperparameters


@dataclass
class PrecomputedAspectRatioEmbedding(Layer):
    """
    Llama 3.2 precomputed aspect ratio embedding.

    Args:
        params: Hyperparameters for this PrecomputedAspectRatioEmbedding layer.
        gate: The gating parameter to control the contribution of the aspect
              ratio embedding.
        embedding: The aspect ratio embedding.
    """

    params: VisionHyperparameters
    gate: TensorValueLike
    embedding: TensorValueLike
    is_gated: bool = False

    def __call__(
        self, hidden_state: TensorValueLike, aspect_ratio_ids: TensorValueLike
    ) -> TensorValue:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(
            (-1, self.params.max_num_tiles, 1, self.params.hidden_size)
        )

        if self.is_gated:
            embeddings = embeddings * ops.tanh(self.gate)

        return hidden_state + embeddings


@dataclass
class PrecomputedPositionEmbedding(Layer):
    """
    Llama 3.2 precomputed position embedding.

    Args:
        params: Hyperparameters for this PrecomputedPositionEmbedding layer.
        gate: The gating parameter to control the contribution of the position
              embedding or another component of the model.
        embedding: The precomputed position embedding.
        tile_embedding: The embedding associated with tiles or patches in a
                        vision model.
    """

    params: VisionHyperparameters
    gate: TensorValueLike
    embedding: TensorValueLike
    tile_embedding: Embedding

    def __call__(
        self, hidden_state: TensorValue, aspect_ratio_ids: TensorValue
    ) -> TensorValue:
        # position embeddings
        gated_position_embedding = (1 - ops.tanh(self.gate)) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.reshape(
            (1, 1, self.params.num_patches, self.params.hidden_size)
        )

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            (
                batch_size,
                self.params.max_num_tiles,
                self.params.num_patches,
                self.params.hidden_size,
            )
        )
        gated_tile_position_embedding = (
            ops.tanh(self.gate) * tile_position_embedding
        )
        return hidden_state + gated_tile_position_embedding
