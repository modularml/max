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
    This is the configuration class to store the configuration of a Llama VisionModel. It is used to instantiate an
    Llama vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama-11B.

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
    dtype: DType = DType.bfloat16
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
    model_type: str = "llama_vision_model"
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
    def max_aspect_ratio_id(self) -> int:
        return len(self.supported_aspect_ratios)

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2 + 1

    @property
    def scale(self):
        return self.hidden_size**-0.5


@dataclass
class TextHyperparameters:
    r"""
    This is the configuration class to store the configuration of a Llama TextModel. It is used to instantiate an
    Llama TextModel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama-11B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Llama TextModel. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling Llama TextModel.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If not
            specified, will default to `num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        cross_attention_layers (`List[int]`, *optional*):
            Indices of the cross attention layers. If not specified, will default to [3, 8, 13, 18, 23, 28, 33, 38].
        dropout (`float`, *optional*, defaults to 0):
            The dropout probability for self- and cross-attention layers.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 128001):
            The id of the end of sentence token.
        pad_token_id (`int`, *optional*, defaults to 128004):
            The id of the padding token.
    """
    dtype: DType = DType.bfloat16
    """The dtype of the weights (is `uint8` for quantized dtypes)."""

    quantization_encoding: QuantizationEncoding | None = None
    """The quantization encoding or `None` if not quantized."""

    # TODO: To be honest I don't think all of them are being used anyway, but
    # let's keep them here for now until the bringup is done then circle back to
    # addressing this technical debt.
    bos_token_id: int = 128000
    cross_attention_layers: list[int] = field(
        default_factory=lambda: [3, 8, 13, 18, 23, 28, 33, 38]
    )
    dropout: float = 0.0
    eos_token_id: int = 128001
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 14336
    max_position_embeddings: int = 131072
    model_type: str = "llama_text_model"
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_key_value_heads: int = 8
    pad_token_id: int = 128004
    rms_norm_eps: float = 1e-05
    rope_scaling: dict[str, float | int | str] = field(
        default_factory=lambda: {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 128256
