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
"""Llama 3.2 vision layer modules."""

from .attention import Attention
from .config import InferenceConfig, SupportedEncodings, SupportedVersions
from .encoder import VisionEncoder, VisionEncoderLayer
from .hyperparameters import TextHyperparameters, VisionHyperparameters
from .llama3_vision import Llama3Vision
from .mlp import MLP
from .positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)
from .vision_model import VisionModel
