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

"""All configurable parameters for Pixtral."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from max.dtype import DType


class SupportedVersions(str, Enum):
    pixtral_12B_2409 = "12B-2409"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class SupportedEncodings(str, Enum):
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def dtype(self) -> DType:
        return _ENCODING_TO_DTYPE[self]

    def hf_model_name(self, version: SupportedVersions) -> str:
        if version == SupportedVersions.pixtral_12B_2409:
            return _ENCODING_TO_MODEL_NAME_PIXTRAL[self]
        else:
            raise ValueError(f"Unsupported version: {version}")


_ENCODING_TO_DTYPE = {
    SupportedEncodings.float32: DType.float32,
    SupportedEncodings.bfloat16: DType.bfloat16,
}


_ENCODING_TO_MODEL_NAME_PIXTRAL = {
    # TODO: add pixtral weights file_name
    SupportedEncodings.float32: "replit-code-v1_5-3b-f32.gguf",
    SupportedEncodings.bfloat16: "replit-code-v1_5-3b-bf16.gguf",
}


from max.dtype import DType
from max.graph.quantization import QuantizationEncoding


# copied from llama for now. should be updated for mistral
@dataclass
class Hyperparameters:
    """All hyperparameters that control the Llama3 model architecture."""

    dtype: DType
    """The dtype of the weights (is `uint8` for quantized dtypes)."""

    quantization_encoding: Optional[QuantizationEncoding] = None
    """The quantization encoding or `None` if not quantized."""

    seq_len: int = 2048
    """Maximum length of the token sequence that can be processed by this model."""

    n_layers: int = 32
    """Number of MultiHeadAttention layers to use in this model."""

    n_heads: int = 32
    """Number of heads for the query to use in the MultiHeadAttention layers."""

    n_kv_heads: int = 8
    """Number of key and value heads to use in the MultiHeadAttention layers."""

    vocab_size: int = 128256
    """Number of tokens in the vocabulary."""

    hidden_dim: int = 4096
    """Hidden dimension of embedded tokens."""

    rope_theta: float = 500000.0
    """Rotary period hyperparameter for rope embeddings."""

    layer_norm_rms_epsilon: float = 1e-5
    """Epsilon value for layer norm calculation."""

    feed_forward_length: int = 500
    """Dimensions in the attention projection layers."""

    has_dedicated_output_weights: bool = True
    """Whether there are dedicated output linear layer weights."""

    @property
    def head_dim(self):
        """Dimension of each head."""
        return self.hidden_dim // self.n_heads

    @property
    def kv_weight_dim(self):
        """Dimension of the key and value attention weights."""
        return self.head_dim * self.n_kv_heads

    @property
    def mask_dtype(self) -> DType:
        """Returns the correct dtype for the model's attention mask.

        When `self.quantization_encoding` is set, then `self.dtype` will simply
        be uint8.
        So in that case we need to pass a float32 attention mask to match the
        activations dtype expected by the quantized CPU flash attention kernel.
        """
        return (
            self.dtype if self.quantization_encoding is None else DType.float32
        )


@dataclass
class LlavaConfig:
    vision_feature_layer: int = -1
    """The index of the layer to select the vision feature."""

    image_token_index: int = 10
    """The image token index to encode the image prompt."""

    vision_feature_select_strategy: str = "full"
    """The feature selection strategy used to select the vision feature from the vision model."""

    image_seq_length: int = 1
    """Sequence length of one image embedding."""


@dataclass
class PixtralVisionHyperparameters:
    """All hyperparameters that control the Pixtral vision model architecture."""

    dtype: DType
    """The dtype of the weights (is `uint8` for quantized dtypes)."""

    quantization_encoding: Optional[QuantizationEncoding] = None
    """The quantization encoding or `None` if not quantized."""

    hidden_dim: int = 1024
    """Dimension of the hidden representations."""

    feed_forward_length = 4096
    """Dimension of the MLP representations."""

    n_layers = 24
    """Number of hidden layers in the Transformer encoder."""

    n_heads = 16
    """Number of attention heads in the Transformer encoder."""

    head_dim = hidden_dim // n_heads

    num_channels = 3
    """Number of input channels in the input images."""

    image_size = 1024
    """Max dimension of the input images."""

    patch_size = 16
    """Size of the image patches."""

    hidden_act = "gelu"
    """Activation function used in the hidden layers."""

    attention_dropout = 0.0
    """Dropout probability for the attention layers."""

    rope_theta = 10000.0
    """The base period of the RoPE embeddings."""

    layer_norm_rms_epsilon = 1e-5

    image_token_id = 10
