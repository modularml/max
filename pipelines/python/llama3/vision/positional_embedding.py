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

"""Llama 3.2 Transformer Vision Model Positional Embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops
from nn.layer import Layer

from .hyperparameters import VisionHyperparameters


# TODO(AIPIPE-133): Implement this.
@dataclass
class PrecomputedAspectRatioEmbedding(Layer):
    """
    Llama 3.2 precomputed aspect ratio embedding.
    """

    params: VisionHyperparameters
    is_gated: bool = False

    def __call__(self, x: TensorValueLike) -> TensorValue:
        raise NotImplementedError("Not implemented yet")


# TODO(AIPIPE-132): Implement this.
@dataclass
class PrecomputedPositionEmbedding(Layer):
    """
    Llama 3.2 precomputed position embedding.
    """

    params: VisionHyperparameters

    def __call__(self, x: TensorValueLike) -> TensorValue:
        raise NotImplementedError("Not implemented yet")
