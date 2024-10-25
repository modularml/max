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

"""Token sampling algorithms."""

from enum import Enum
from typing import Optional

from max.dtype import DType
from max.graph import Dim, Graph, Shape, TensorType, ops


def token_sampler(top_k: Optional[int], dtype: DType):
    logits_type = TensorType(dtype, ["batch", "vocab_size"])
    with Graph("token_sampler", input_types=[logits_type]) as graph:
        (logits, *_) = graph.inputs
        if top_k is not None:
            shape = Shape(logits.shape)
            shape[-1] = Dim(1)
            tokens = ops.custom(
                "topk_fused_sampling",
                [ops.constant(top_k, dtype=DType.int64), logits],
                [TensorType(DType.int64, shape)],
            )[0]
        else:
            tokens = ops.argmax(logits)
        graph.output(tokens)
        return graph


def argmax_sampler(dtype: DType):
    logits_type = TensorType(dtype, ["batch", "vocab_size"])
    return Graph("argmax", ops.argmax, input_types=[logits_type])
