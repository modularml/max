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
from typing import List


import numpy as np
from max.graph import ops


def causal_attention_mask_2d_from_imgs(
    imgs: List[np.ndarray], patch_size, batch_size, fill_val=-10000.0
):
    """
    imgs: list of images of shape = (height, width, num_channels)

    It is list representing the sizes of different blocks of patch embeddings.
    patch_embeds: embeddings of each patch. tensor shape = [batch_size, num_patches, hidden_size]
    """
    # generate list of (num_patches_in_height * num_patches_in_width) for each image
    num_patches_list = [
        img.shape[0] // patch_size * img.shape[1] // patch_size for img in imgs
    ]

    # seq_length is number of patches in all images
    seq_len = sum(num_patches_list)
    mask_shape = (seq_len, seq_len)

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_matrix = np.full(mask_shape, fill_val, dtype=np.float32)

    # block_end_idx and block_start_idx are calculated using cumulative sums of
    # patch_embeds_list. These indicate the starting and ending indices of each
    # block of embeddings.
    block_end_idx = np.cumsum(num_patches_list)
    block_start_idx = np.cumsum(np.concatenate(([0], num_patches_list[:-1])))

    # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
    for start, end in zip(block_start_idx, block_end_idx):
        fill_matrix[int(start) : int(end), int(start) : int(end)] = 0

    # Expand the mask dimensions to match the expected input shape
    fill_matrix = np.expand_dims(fill_matrix, axis=(0, 1))  # Add two new axes
    fill_matrix = np.broadcast_to(
        fill_matrix, (batch_size, 1, seq_len, seq_len)
    )
    return fill_matrix


def causal_attention_mask_2d(num_patches_list, patch_embeds):
    """
    num_patches_list: list of (num_patches_in_height * num_patches_in_width) for each image
    It is list representing the sizes of different blocks of patch embeddings.
    patch_embeds: embeddings of each patch. tensor shape = [batch_size, num_patches, hidden_size]
    """
    seq_len = int(patch_embeds.shape[1])
    mask_shape = (seq_len, seq_len)

    # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    fill_matrix = np.full(mask_shape, fill_val, dtype=np.float32)

    # block_end_idx and block_start_idx are calculated using cumulative sums of
    # patch_embeds_list. These indicate the starting and ending indices of each
    # block of embeddings.
    block_end_idx = np.cumsum(num_patches_list)
    block_start_idx = np.cumsum(np.concatenate(([0], num_patches_list[:-1])))

    # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
    for start, end in zip(block_start_idx, block_end_idx):
        fill_matrix[int(start) : int(end), int(start) : int(end)] = 0

    # Expand the mask dimensions to match the expected input shape
    fill_matrix = np.expand_dims(fill_matrix, axis=(0, 1))  # Add two new axes
    fill_matrix = np.broadcast_to(
        fill_matrix, (int(patch_embeds.shape[0]), 1, seq_len, seq_len)
    )
    return fill_matrix


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : int(x.shape[-1]) // 2]
    x2 = x[..., int(x.shape[-1]) // 2 :]
    return ops.concat((-x2, x1), axis=-1)
