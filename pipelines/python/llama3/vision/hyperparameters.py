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
"""
Llama3.2 vision model hyperparameters. These may be combined / consolidated
under pipelines/python/llama3/model/hyperparameters.py in the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding


@dataclass
class VisionHyperparameters:
    r"""
    This is the configuration class to store the configuration of a [`MllamaVisionModel`]. It is used to instantiate an
    Mllama vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-11B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_global_layers (`int`, *optional*, defaults to 8):
            Number of global layers in the Transformer encoder.
            Vision model has a second transformer encoder, called global.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        vision_output_dim (`int`, *optional*, defaults to 7680):
            Dimensionality of the vision model output. Includes output of transformer
            encoder with intermediate layers and global transformer encoder.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image *tile*.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        max_num_tiles (`int`, *optional*, defaults to 4):
            Maximum number of tiles for image splitting.
        intermediate_layers_indices (`List[int]`, *optional*, defaults to [3, 7, 15, 23, 30]):
            Indices of intermediate layers of transformer encoder from which to extract and output features.
            These output features are concatenated with final hidden state of transformer encoder.
        supported_aspect_ratios (`List[List[int]]`, *optional*):
            List of supported aspect ratios for image splitting. If not specified, the default supported aspect ratios
            are [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]] for `max_num_tiles=4`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """
    dtype: DType
    """The dtype of the weights (is `uint8` for quantized dtypes)."""

    quantization_encoding: QuantizationEncoding | None = None
    """The quantization encoding or `None` if not quantized."""

    # TODO: To be honest I don't think all of them are being used anyway, but
    # let's keep them here for now until the bringup is done then circle back to
    # addressing this technical debt.
    attention_heads: int = 16
    hidden_size: int = 1280
    image_size: int = 448
    initializer_range: float = 0.02
    intermediate_layers_indices: list[int] = field(
        default_factory=lambda: [3, 7, 15, 23, 30]
    )
    intermediate_size: int = 5120
    max_num_tiles: int = 4
    model_type: str = "mllama_vision_model"
    norm_eps: float = 1e-05
    num_channels: int = 3
    num_global_layers: int = 8
    num_hidden_layers: int = 32
    patch_size: int = 14
    supported_aspect_ratios: list[list[int]] = field(
        default_factory=lambda: [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [3, 1],
            [4, 1],
        ]
    )
    vision_output_dim: int = 7680

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2 + 1

    @property
    def scale(self):
        return self.hidden_size**-0.5
