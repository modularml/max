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
"""Llama3 model hyperparameters."""

import os
from dataclasses import dataclass
from typing import Optional

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding


@dataclass
class Hyperparameters:
    dtype: DType
    quantization_encoding: Optional[QuantizationEncoding] = None

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

    force_naive_kv_cache: bool = False
    """Force using the naive KV cache even for configs supporting the opaque KV cache."""

    @property
    def head_dim(self):
        """Dimension of each head."""
        return self.hidden_dim // self.n_heads

    @property
    def kv_weight_dim(self):
        """Dimension of the key and value attention weights."""
        return self.head_dim * self.n_kv_heads

    @property
    def use_opaque(self):
        """Boolean to use opaque kv caching optimizations."""
        return (
            self.quantization_encoding is None
            and "TMP_USE_OPAQUE" in os.environ
            and not self.force_naive_kv_cache
        )
