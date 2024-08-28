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
"""The rope embedding used within the model."""


from dataclasses import dataclass
from typing import Optional

import numpy as np
from max.graph import DimLike, TensorValue, ValueLike, ops


@dataclass
class RotaryEmbedding:
    """
    RotaryEmbedding layer to calculate and apply the frequency tensor for complex exponentials.
    """

    freqs_cis: ValueLike

    def __init__(
        self,
        dim: DimLike,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        rope_scaling: Optional[np.array] = None,
    ):
        self.freqs_cis = self.calculate_freqs_cis(
            dim, n_heads, theta, max_seq_len, rope_scaling
        )

    def calculate_freqs_cis(
        self,
        dim: DimLike,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        rope_scaling: Optional[np.array],
    ) -> TensorValue:
        """
        Computes the frequency tensor for complex exponentials (cis)
        for a given seq_len. Tensor is scaled with theta parameter.
        Required to apply Rotary Position Embedding (RoPE) to tensor.
        See 'Roformer: Enhanced Transformer with Rotary Embedding'
        (arxiv.org/pdf/2104.09864).

        Args:
            max_seq_len: The maximum sequence length for model's input.
            theta: Hyperparameter used to control the frequency scaling of the sinusoidal components of the embeddings.
            rope_scaling: Scaling factor for the positional frequencies.

        Returns:
            The frequency tensor for complex exponentials with shape
                (max_seq_len * 2, dim//(2 * n_heads), 2)
        """
        n = dim // n_heads
        # TODO (MSDK-655): Use ops.arange() here when implemented.
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        iota = np.arange(0, n - 1, 2, dtype=np.float64)
        if rope_scaling is not None:
            iota = iota * rope_scaling.astype(np.float64)
        freqs = (1.0 / (theta ** (iota / n))).astype(np.float32)
        # TODO (MSDK-655): Use ops.arange() here when implemented.
        t = np.arange(0, max_seq_len * 2.0, dtype=np.float32)
        freqs = ops.outer(t, freqs)
        return ops.stack([ops.cos(freqs), ops.sin(freqs)], axis=-1)

    def __call__(
        self, x: ValueLike, start_pos: int, seq_len: int
    ) -> TensorValue:
        """Applies rotary positional embeddings (RoPE) to `x`.

        Args:
            x: Activation tensor with shape (batch, seq_len, n_kv_heads, head_dim).
            start_pos: starting position of input tensor
            seq_len: length of input tensor

        Returns:
            Input activation tensor with rotary positional embeddings applied and
            the same shape as `x`.
        """
        v = TensorValue(x)

        complex = ops.as_interleaved_complex(v)

        start_pos_val = TensorValue.from_dim(start_pos)
        seq_len_val = TensorValue.from_dim(seq_len)
        freqs_cis_sliced = self.freqs_cis[
            (slice(start_pos_val, start_pos_val + seq_len_val), seq_len),
        ]

        freqs_cis_bcast = ops.unsqueeze(ops.unsqueeze(freqs_cis_sliced, 1), 0)

        x_re = complex[..., 0]
        x_im = complex[..., 1]

        freqs_re = freqs_cis_bcast[..., 0]
        freqs_im = freqs_cis_bcast[..., 1]

        rope_re = (x_re * freqs_re) - (x_im * freqs_im)
        rope_im = (x_re * freqs_im) + (x_im * freqs_re)

        rope_complex = ops.stack([rope_re, rope_im], axis=-1)

        return ops.reshape(rope_complex, v.shape)
