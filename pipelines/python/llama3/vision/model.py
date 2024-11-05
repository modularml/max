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

from max.dtype import DType
from max.graph import Graph, TensorValue, TensorValueLike, ops
from max.graph.weights import SafetensorWeights
from nn import Conv2D, Embedding, Linear, LPLayerNorm
from nn.layer import Layer

from .class_embedding import ClassEmbedding
from .encoder import VisionEncoder, VisionEncoderLayer
from .hyperparameters import VisionHyperparameters
from .mlp import MLP
from .positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)


def lp_layer_norm(
    dtype: DType,
    size: int,
    eps: float,
    weights: SafetensorWeights,
) -> LPLayerNorm:
    """
    Helper function to instantiate a LPLayerNorm layer.
    """
    return LPLayerNorm(weights.weight.allocate(dtype, [size]), eps=eps)


# TODO: Copy pasted from other pipelines - maybe worth moving to a util subdir?
def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    """
    Helper function to instantiate a Linear layer.
    """
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


@dataclass
class VisionModel(Layer):
    """
    Llama 3.2 vision model which consists of two vision encoders.
    """

    # TODO: Optional (|) fields are stubbed out for now.
    # Some of them are already implemented but None can't be removed yet
    # as we need to update some unit tests - to be addressed in a future PR.
    params: VisionHyperparameters
    patch_embedding: Conv2D
    gated_positional_embedding: PrecomputedPositionEmbedding | None = None
    pre_tile_positional_embedding: PrecomputedAspectRatioEmbedding | None = None
    post_tile_positional_embedding: PrecomputedAspectRatioEmbedding | None = (
        None
    )
    layernorm_pre: LPLayerNorm | None = None
    layernorm_post: LPLayerNorm | None = None
    class_embedding: ClassEmbedding | None = None
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
) -> VisionModel:
    gated_positional_embedding = PrecomputedPositionEmbedding(
        params=params,
        gate=weights.vision_model.gated_positional_embedding.gate.allocate(
            DType.bfloat16, [1]
        ),
        embedding=weights.vision_model.gated_positional_embedding.embedding.allocate(
            DType.bfloat16, [params.num_patches, params.hidden_size]
        ),
        tile_embedding=Embedding(
            weights.vision_model.gated_positional_embedding.tile_embedding.weight.allocate(
                DType.bfloat16,
                [
                    params.max_aspect_ratio_id + 1,
                    params.max_num_tiles
                    * params.num_patches
                    * params.hidden_size,
                ],
            ),
        ),
    )

    pre_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
        params=params,
        gate=weights.vision_model.pre_tile_positional_embedding.gate.allocate(
            DType.bfloat16, [1]
        ),
        embedding=Embedding(
            weights.vision_model.pre_tile_positional_embedding.embedding.weight.allocate(
                DType.bfloat16,
                [
                    params.max_aspect_ratio_id + 1,
                    params.max_num_tiles * params.hidden_size,
                ],
            ),
        ),
        is_gated=True,
    )

    post_tile_positional_embedding = PrecomputedAspectRatioEmbedding(
        params=params,
        gate=weights.vision_model.post_tile_positional_embedding.gate.allocate(
            DType.bfloat16, [1]
        ),
        embedding=Embedding(
            weights.vision_model.post_tile_positional_embedding.embedding.weight.allocate(
                DType.bfloat16,
                [
                    params.max_aspect_ratio_id + 1,
                    params.max_num_tiles * params.hidden_size,
                ],
            ),
        ),
        is_gated=True,
    )

    # patch_embedding filter has a filter of [1280, 3, 14, 14]
    patch_embedding = Conv2D(
        filter=weights.vision_model.patch_embedding.weight.allocate(
            DType.bfloat16,
            [
                params.hidden_size,
                params.num_channels,
                params.patch_size,
                params.patch_size,
            ],
        ),
        stride=params.patch_size,
        padding=(0, 0, 0, 0),
        bias=False,
    )

    class_embedding = weights.vision_model.class_embedding.allocate(
        DType.bfloat16, [params.hidden_size]
    )

    layernorm_pre = lp_layer_norm(
        dtype=DType.bfloat16,
        size=params.hidden_size,
        eps=params.norm_eps,
        weights=weights.vision_model.layernorm_pre,
    )

    layernorm_post = lp_layer_norm(
        dtype=DType.bfloat16,
        size=params.hidden_size,
        eps=params.norm_eps,
        weights=weights.vision_model.layernorm_post,
    )

    transformer_encoder_layers: list[VisionEncoderLayer] = []

    for index in range(params.num_hidden_layers):
        curr_layer_weight = weights.vision_model.transformer.layers[index]
        transformer_encoder_layers.append(
            VisionEncoderLayer(
                mlp=MLP(
                    linear(
                        dtype=DType.bfloat16,
                        in_features=params.intermediate_size,
                        out_features=params.hidden_size,
                        weights=curr_layer_weight.mlp.fc1,
                    ),
                    linear(
                        dtype=DType.bfloat16,
                        in_features=params.hidden_size,
                        out_features=params.intermediate_size,
                        weights=curr_layer_weight.mlp.fc2,
                    ),
                ),
                input_layernorm=lp_layer_norm(
                    dtype=DType.bfloat16,
                    size=params.hidden_size,
                    eps=params.norm_eps,
                    weights=curr_layer_weight.input_layernorm,
                ),
                post_attention_layernorm=lp_layer_norm(
                    dtype=DType.bfloat16,
                    size=params.hidden_size,
                    eps=params.norm_eps,
                    weights=curr_layer_weight.post_attention_layernorm,
                ),
                is_gated=False,
                gate_attn=None,
                gate_ffn=None,
            )
        )
    transformer = VisionEncoder(transformer_encoder_layers)

    global_transformer_layers: list[VisionEncoderLayer] = []

    for index in range(params.num_global_layers):
        curr_layer_weight = weights.vision_model.global_transformer.layers[
            index
        ]

        global_transformer_layers.append(
            VisionEncoderLayer(
                mlp=MLP(
                    linear(
                        dtype=DType.bfloat16,
                        in_features=params.intermediate_size,
                        out_features=params.hidden_size,
                        weights=curr_layer_weight.mlp.fc1,
                    ),
                    linear(
                        dtype=DType.bfloat16,
                        in_features=params.hidden_size,
                        out_features=params.intermediate_size,
                        weights=curr_layer_weight.mlp.fc2,
                    ),
                ),
                input_layernorm=lp_layer_norm(
                    dtype=DType.bfloat16,
                    size=params.hidden_size,
                    eps=params.norm_eps,
                    weights=curr_layer_weight.input_layernorm,
                ),
                post_attention_layernorm=lp_layer_norm(
                    dtype=DType.bfloat16,
                    size=params.hidden_size,
                    eps=params.norm_eps,
                    weights=curr_layer_weight.post_attention_layernorm,
                ),
                is_gated=True,
                gate_attn=curr_layer_weight.gate_attn.allocate(
                    DType.bfloat16, [1]
                ),
                gate_ffn=curr_layer_weight.gate_ffn.allocate(
                    DType.bfloat16, [1]
                ),
            )
        )
    global_transformer = VisionEncoder(global_transformer_layers)

    return VisionModel(
        params=params,
        gated_positional_embedding=gated_positional_embedding,
        pre_tile_positional_embedding=pre_tile_positional_embedding,
        post_tile_positional_embedding=post_tile_positional_embedding,
        patch_embedding=patch_embedding,
        class_embedding=class_embedding,
        layernorm_pre=layernorm_pre,
        layernorm_post=layernorm_post,
        transformer=transformer,
        global_transformer=global_transformer,
    )
