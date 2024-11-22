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

"""All configurable parameters for Pixtral."""

from max.pipelines import HuggingFaceFile, SupportedEncoding


def get_pixtral_huggingface_file(
    encoding: SupportedEncoding,
) -> HuggingFaceFile:
    if encoding == SupportedEncoding.float32:
        raise ValueError(f"Pixtral does not support: {encoding}")
    elif encoding == SupportedEncoding.bfloat16:
        # using official mistral weights
        return HuggingFaceFile(
            "mistralai/Pixtral-12B-2409", "consolidated.safetensors"
        )
    else:
        raise ValueError(f"Pixtral does not support: {encoding}")
