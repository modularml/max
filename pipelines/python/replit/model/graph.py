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

from max.graph import Graph
from max.graph.weights import GGUFWeights
from nn.kv_cache.cache_params import KVCacheParams
from .hyperparameters import Hyperparameters


def _build_graph(
    hyperparameters: Hyperparameters,
    weights: GGUFWeights,
    kv_params: KVCacheParams,
) -> Graph:
    raise NotImplementedError("replit graph not implemented.")
