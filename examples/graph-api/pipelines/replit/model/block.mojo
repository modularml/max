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
"""The core Transformer block of the model."""

from collections import Optional

from max.graph import ops, Symbol, Graph
from pathlib import Path
from pipelines.weights.gguf import GGUFFile
from pipelines.weights.loadable_model import LoadableModel

from pipelines.nn import Linear
from ..model.norm import LPLayerNorm
from ..weights.hyperparams import HyperParams


@value
struct MPTMLP:
    """Multiplayer perceptron used in MPT."""

    var up_proj: Linear
    var down_proj: Linear

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        with g.layer("MPTMLP"):
            return self.down_proj(ops.gelu(self.up_proj(input)))
