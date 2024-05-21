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
from collections import Optional, List
from pathlib import Path

from max.graph import ops, Dim, TensorType, Symbol


@value
struct Embedding:
    """A layer that converts tokens into dense vectors."""

    var weights: Symbol

    def __call__(self, input: Symbol) -> Symbol:
        return ops.gather(self.weights, input, axis=0)


@value
struct SharedEmbedding:
    """An embedding layer that can both embed and unembed inputs."""

    var weights: Symbol

    def __call__(self, input: Symbol, unembed: Bool = False) -> Symbol:
        if unembed:
            return input @ ops.transpose_matrix(self.weights)
        return ops.gather(self.weights, input, axis=0)
