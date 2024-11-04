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

"""All configurable parameters for Replit."""

from max.pipelines import SupportedEncoding, HuggingFaceFile


def get_replit_huggingface_file(encoding: SupportedEncoding) -> HuggingFaceFile:
    if encoding == SupportedEncoding.float32:
        return HuggingFaceFile(
            "modularai/replit-code-1.5",
            "replit-code-v1_5-3b-f32.gguf",
        )
    elif encoding == SupportedEncoding.bfloat16:
        return HuggingFaceFile(
            "modularai/replit-code-1.5",
            "replit-code-v1_5-3b-f32.gguf",
        )
    else:
        raise ValueError(f"replit does not support: {encoding}")
