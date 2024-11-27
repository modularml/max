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

from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import KVCacheManager, KVCacheParams

# from mistral.model.graph import _transformer
from nn import Linear

from ..llava.llava import LlavaConditionalGeneration
from ..llava.llava_projector import LlavaMultiModalConnector
from ..vision_encoder.graph import _vision_encoder
from .mistral_graph import _transformer


def _linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    """Unlike the vision encoder's version, this linear layer has a bias.
    This linear layer is used by the LlavaMultiModalConnector
    """
    # TODO: How to init bias?
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None),
        bias=weights.bias.allocate(dtype, [in_features], None),
    )


def _multi_modal_projector(
    dtype: DType,
    params: PipelineConfig,
    weights: SafetensorWeights,
) -> LlavaMultiModalConnector:
    """Connects the vision encoder to the text decoder.
    This MLP projects the patch embeddings to the text-encoder's embeddings space.
    Input shape:
    Output shape:
    """
    return LlavaMultiModalConnector(
        _linear(
            dtype,
            params.huggingface_config.text_config.hidden_size,
            params.huggingface_config.vision_config.hidden_size,
            weights.linear_1,
        ),
        _linear(
            dtype,
            params.huggingface_config.text_config.hidden_size,
            params.huggingface_config.text_config.hidden_size,
            weights.linear_2,
        ),
    )


def _llava(
    graph: Graph,
    params: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
) -> LlavaConditionalGeneration:
    # params for vision encoder in pixtral config.json are under vision_config.
    # vision encoder params missing from pixtral config.json:
    # num_attention_heads, num_channels, hidden_size, attention_dropout, intermediate_size, num_hidden_layers

    vision_encoder = _vision_encoder(graph, params, weights)
    multi_modal_projector = _multi_modal_projector(
        params.dtype, params, weights.multi_modal_projector
    )
    # Weights of pixtral have the same names and shapes as weights of mistral.
    language_model = _transformer(graph, params, weights, kv_params)

    pad_token_id = -1
    return LlavaConditionalGeneration(
        vision_encoder,
        multi_modal_projector,
        language_model,
        params.huggingface_config.text_config.vocab_size,
        pad_token_id,
        params.huggingface_config.image_token_index,
        params.huggingface_config.vision_feature_layer,
        params.huggingface_config.vision_feature_select_strategy,
        params.huggingface_config.image_seq_length,
    )


def _build_graph(
    params: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    # Graph input types.
    kv_cache_args = kv_manager.input_symbols()

    # TODO: Do we need text_token_type and input_row_offset_type from mistral?
    # TODO: What is the input type to Pixtral?
    input_ids_type = TensorType(
        # DType.int64, ["batch_size", "sequence_length"]
        DType.int64,
        [1, 502],
    )
    pixel_values_type = TensorType(
        DType.bfloat16,
        [304, 400, 3],
        #                ["height", "width", "num_channels"],
    )
    input_row_offset_type = TensorType(
        DType.uint32, shape=["input_row_offset_len"]
    )

    # TODO: Use symbolic dims.
    # Initialize Graph.
    with Graph(
        "pixtral",
        input_types=[
            input_ids_type,
            pixel_values_type,
            input_row_offset_type,
            *kv_cache_args,  # type: ignore
        ],
    ) as graph:
        model = _llava(graph, params, weights, kv_params)
        input_ids, pixel_values, input_row_offset, *kv_cache = graph.inputs
        logits = model(
            input_ids,  # type: ignore
            pixel_values,  # type: ignore
            kv_cache,  # type: ignore
            input_row_offset=input_row_offset,
        )
        graph.output(logits)
        return graph
