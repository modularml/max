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
"""All configurable parameters for Llama3."""

from max.pipelines import HuggingFaceFile, SupportedEncoding


def get_coder_huggingface_files(
    version: str, encoding: SupportedEncoding
) -> list[HuggingFaceFile]:
    if version == "1.5":
        if encoding == SupportedEncoding.bfloat16:
            return [
                HuggingFaceFile(
                    "deepseek-ai/deepseek-coder-7b-instruct-v1.5", f
                )
                for f in [
                    "model-00001-of-00003.safetensors",
                    "model-00002-of-00003.safetensors",
                    "model-00003-of-00003.safetensors",
                ]
            ]

        else:
            raise ValueError(f"encoding does not hf file: {encoding}")

    else:
        raise ValueError(f"version {version} not supported for llama")
