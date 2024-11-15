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

"""Llama 3.2 Transformer Vision Language Model cross attention decoder."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.graph.weights import SafetensorWeights
from nn import MLP, Linear, RMSNorm
from nn.layer import Layer

from .cache import Cache
from .hyperparameters import TextHyperparameters


# TODO(MAXCORE-145): Implement this!
@dataclass
class SelfAttentionDecoderLayer:
    pass
