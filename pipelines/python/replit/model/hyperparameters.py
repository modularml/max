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
"""Replit model hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding

from ..config import InferenceConfig


@dataclass
class Hyperparameters:
    dtype: DType
    quantization_encoding: Optional[QuantizationEncoding] = None

    seq_len: int = 512
    """Maximum length of the token sequence that can be processed by this model."""

    num_layers: int = 32
    """Number of MultiHeadAttention layers to use in this model."""

    n_heads: int = 24
    """Number of heads for the query to use in the MultiHeadAttention layers."""

    n_kv_heads: int = 8
    """Number of key and value heads to use in the MultiHeadAttention layers."""

    vocab_size: int = 32768
    """Number of tokens in the vocabulary."""

    hidden_dim: int = 3072
    """Hidden dimension of embedded tokens."""

    layer_norm_epsilon: float = 1e-5
    """Epsilon value for layer norm calculation."""

    casual: bool = True
    """A toggle for adjusting the stylistic output of Replit, if true the model generates
    code in a more relaxed and informal style, allowing for greater flexibility and 
    creativity in code generation. If false, the model produces more formal and structured 
    code outputs."""

    alibi: bool = True
    """Hyperparameter that enables the model to apply linear biases to attention scores.
    This approach allows the model to maintain contextual relevant over extented input
    lengths, enhancing its performance in tasks requiring long-range dependencies.
    """

    alibi_bias_max: int = 8
    """The maximum bias applied in ALIBI. This hyperparameter controls the extent of 
    biasing in attention calculations, allowing for fine-tuning of how strongly the model
    prioritizes certain tokens based on their position within the input sequence.
    """

    @classmethod
    def load(cls, config: InferenceConfig, **kwargs):
        # Update kwargs based on config.
        if "dtype" not in kwargs:
            kwargs["dtype"] = config.quantization_encoding.dtype

        if "quantization_encoding" not in kwargs:
            kwargs[
                "quantization_encoding"
            ] = config.quantization_encoding.quantization_encoding

        if "seq_len" not in kwargs:
            kwargs["seq_len"] = config.seq_len

        return cls(**kwargs)

    @property
    def head_dim(self):
        """Dimension of each head."""
        return self.hidden_dim // self.n_heads
