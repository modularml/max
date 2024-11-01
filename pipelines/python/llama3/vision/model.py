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

"""Llama 3.2 Transformer Vision Model."""

from __future__ import annotations
from dataclasses import dataclass

from max.graph import Graph, TensorValue, TensorValueLike, ops
from max.graph.weights import SafetensorWeights
from nn import Conv2D, LPLayerNorm
from nn.layer import Layer
from .class_embedding import ClassEmbedding
from .encoder import VisionEncoder
from .hyperparameters import VisionHyperparameters
from .positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)


@dataclass
class VisionModel(Layer):
    """
    Llama 3.2 vision model which consists of two vision encoders.
    """

    # TODO: Optional (|) fields are stubbed out for now.
    params: VisionHyperparameters
    gated_positional_embedding: PrecomputedPositionEmbedding
    pre_tile_positional_embedding: PrecomputedAspectRatioEmbedding
    post_tile_positional_embedding: PrecomputedAspectRatioEmbedding
    patch_embedding: Conv2D | None = None
    class_embedding: ClassEmbedding | None = None
    layernorm_pre: LPLayerNorm | None = None
    layernorm_post: LPLayerNorm | None = None
    transformer: VisionEncoder | None = None
    global_transformer: VisionEncoder | None = None

    def __call__(
        self,
        pixel_values: TensorValueLike,
        aspect_ratio_ids: TensorValueLike,
        aspect_ratio_mask: TensorValueLike,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[TensorValue]:
        raise NotImplementedError("Not implemented yet")
        return (pixel_values,)


def instantiate_vision_model(
    graph: Graph,
    params: VisionHyperparameters,
    weights: SafetensorWeights,
):
    raise NotImplementedError("Not implemented yet")
