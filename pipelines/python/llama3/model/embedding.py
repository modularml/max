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

import enum
from dataclasses import dataclass
from typing import Optional

from max.graph import TensorValue, ValueLike, ops
from max.graph.quantization import QuantizationEncoding


@dataclass
class Embedding:
    weights: TensorValue
    quantization_encoding: Optional[QuantizationEncoding] = None

    def __call__(self, indices: ValueLike) -> TensorValue:
        result = ops.gather(self.weights, indices, axis=0)
        if self.quantization_encoding is not None:
            result = ops.dequantize(self.quantization_encoding, result)
        return result
