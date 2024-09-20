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

from __future__ import annotations

import numpy as np


def causal_attention_mask(
    start_pos: list[int], seq_len: list[int]
) -> np.ndarray:
    # Each example in the batch has a "start position", which is the length
    # of the previously encoded tokens ("context"), and a "sequence length",
    # which is the number of additional tokens to be encoded in this pass.
    #
    # "Causal attention" means that each token can "see" tokens before it.
    # The attention layer adds the mask to the attention scores and then
    # performs a softmax, so for tokens that a given token can "see" the mask
    # wants to produce a 0, meaning to pass the attention through as normal,
    # and for tokens that can't be "seen" the mask should produce -inf, which
    # will result in them being functionally ignored after the softmax operation.
    #
    # We call the total length "post_seq_len", referring to the total context
    # length after this pass concludes.
    start_pos: np.ndarray = np.array(start_pos, dtype=np.int64)
    seq_len: np.ndarray = np.array(seq_len, dtype=np.int64)
    post_seq_len = start_pos + seq_len

    # Mask shape: for each token being generated, attend to tokens _before_ it
    # in the entire sequence including context. Pad all values to the longest
    # sequence length and total length.
    mask_shape = (seq_len.max(), post_seq_len.max())
    return np.stack(
        [np.triu(np.full(mask_shape, float("-inf")), k=k) for k in start_pos]
    )
