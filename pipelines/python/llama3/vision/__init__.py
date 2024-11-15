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
from .conditional_generator import ConditionalGenerator
from .config import InferenceConfig, SupportedEncodings, SupportedVersions
from .cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)
from .encoder import VisionEncoder, VisionEncoderLayer
from .hyperparameters import TextHyperparameters, VisionHyperparameters
from .language_model import CausalLanguageModel
from .llama3_vision import Llama3Vision
from .mlp import MLP
from .positional_embedding import (
    PrecomputedAspectRatioEmbedding,
    PrecomputedPositionEmbedding,
)
from .self_attention_decoder import SelfAttentionDecoderLayer
from .vision_model import VisionModel
