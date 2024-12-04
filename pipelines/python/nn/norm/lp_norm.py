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

"""Layer Normalization layer."""

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops

from ..layer import Layer


@dataclass
class LPLayerNorm(Layer):
    """Layer normalization block."""

    weight: TensorValueLike
    eps: float = 1e-6

    def __call__(self, input: TensorValue):
        # TODO: AIPIPE-95 Replace with a broadcasting rmo.layer_norm
        return ops.layer_norm(
            input,
            ops.cast(self.weight, input.dtype),
            ops.broadcast_to(
                ops.constant(0.0, input.dtype),
                shape=(input.shape[-1],),
            ),
            self.eps,
        )
