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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class SupportedEncodings(str, Enum):
    float32 = "float32"
    bfloat16 = "bfloat16"
    q4_0 = "q4_0"
    q4_k = "q4_k"
    q6_k = "q6_k"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class SupportedVersions(str, Enum):
    llama3 = "3"
    llama3_1 = "3.1"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


PRETRAINED_MODEL_WEIGHTS = {
    SupportedVersions.llama3: {
        SupportedEncodings.q4_0: "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        SupportedEncodings.q4_k: "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        SupportedEncodings.q6_k: "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf",
        SupportedEncodings.bfloat16: "https://huggingface.co/ddh0/Meta-Llama-3-8B-Instruct-bf16-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-bf16.gguf",
        SupportedEncodings.float32: "https://huggingface.co/brendanduke/Llama-3-8B-f32.gguf/resolve/main/llama3-8b-f32.gguf",
    },
    SupportedVersions.llama3_1: {
        SupportedEncodings.q4_0: "https://huggingface.co/kaetemi/Meta-Llama-3.1-8B-Q4_0-GGUF/resolve/main/meta-llama-3.1-8b-q4_0.gguf",
        SupportedEncodings.q4_k: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        SupportedEncodings.q6_k: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        SupportedEncodings.bfloat16: "https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-bf16.gguf",
        SupportedEncodings.float32: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-f32.gguf",
    },
}


@dataclass
class InferenceConfig:
    weight_path: Optional[Union[str, Path]] = None
    """Path or URL of the model weights."""

    version: SupportedVersions = SupportedVersions.llama3_1
    """Llama version."""

    quantization_encoding: SupportedEncodings = SupportedEncodings.q4_k
    """Quantization encoding type."""

    serialized_model_path: Optional[Union[str, Path]] = None
    """If specified, tries to load a serialized model from this path."""

    max_length: int = 512
    """Controls the maximum length of the text sequence (includes the input tokens)."""

    max_new_tokens: int = -1
    """Controls the maximum length of the text sequence (does not include the input tokens)."""

    batch_size: int = 1
    """Batch size of inputs to the model."""

    @staticmethod
    def help():
        return {
            "weight_path": (
                "Path or URL of the model weights. If not provided, default"
                " weights will be downloaded based on the version and"
                " quantization encoding."
            ),
            "version": "Llama version.",
            "quantization_encoding": (
                "The encoding to use for a datatype that can be quantized to a"
                " low bits per weight format."
            ),
            "max_length": (
                "Controls the maximum length of the text sequence (includes the"
                " input tokens)."
            ),
            "version": (
                "Controls the maximum length of the text sequence (does not"
                " include the input tokens)."
            ),
            "batch_size": "Batch size of inputs to the model.",
        }

    def remote_weight_location(self):
        if self.weight_path is not None:
            return self.weight_path
        try:
            version_weights = PRETRAINED_MODEL_WEIGHTS[self.version]
        except KeyError:
            raise ValueError(f"unsupported model version {self.version}")
        try:
            return version_weights[self.quantization_encoding]
        except KeyError:
            raise ValueError(
                f"quantization of {self.quantization_encoding} not"
                f" supported for version {self.version}"
            )
