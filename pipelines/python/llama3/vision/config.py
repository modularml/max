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
"""
All configurable parameters for Llama 3.2 vision model.
This should eventually be merged with the parent Llama config.py but
since we have ongoing refactoring for PipelineConfig, we'll do this later on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from max.driver import DeviceSpec
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from nn.kv_cache import KVCacheStrategy


class SupportedVersions(str, Enum):
    llama3_2 = "3.2"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class SupportedEncodings(str, Enum):
    bfloat16 = "bfloat16"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def quantization_encoding(self) -> QuantizationEncoding | None:
        return None

    @property
    def dtype(self) -> DType:
        return _ENCODING_TO_DTYPE[self]

    def hf_model_name(self, version: SupportedVersions) -> str | list[str]:
        if version == SupportedVersions.llama3_2:
            return _ENCODING_TO_MODEL_NAME_LLAMA3_2[self]
        else:
            raise ValueError(f"Unsupported version: {version}")


_ENCODING_TO_DTYPE = {
    SupportedEncodings.bfloat16: DType.bfloat16,
}

_ENCODING_TO_MODEL_NAME_LLAMA3_2 = {
    SupportedEncodings.bfloat16: [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]
}


@dataclass
class InferenceConfig:
    device_spec: DeviceSpec = DeviceSpec.cpu()
    """Device to run inference upon."""

    weight_path: str | Path | None = None
    """Path or URL of the model weights."""

    huggingface_weights: str | list[str] | None = field(
        default_factory=lambda: [
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ]
    )
    """Hugging Face weights to download and use with this model."""

    version: SupportedVersions = SupportedVersions.llama3_2
    """Llama 3.2 version."""

    # TODO: Llama 3.2 vision only supports "bfloat16", so this field doesn't
    # need to be configurable by the user.
    quantization_encoding: SupportedEncodings | None = (
        SupportedEncodings.bfloat16
    )
    """Encoding type."""

    serialized_model_path: str | Path | None = None
    """If specified, tries to load a serialized model from this path."""

    save_to_serialized_model_path: str | Path | None = None
    """If specified, tries to save a serialized model to this path."""

    repo_id: str = "meta-llama/Llama-3.2-11B-Vision"
    """Repo ID to use HF weights and tokenizers."""

    @staticmethod
    def help():
        return {
            "weight_path": (
                "Path or URL of the model weights. If not provided, default"
                " weights will be downloaded based on the version and"
                " quantization encoding."
            ),
            "huggingface_weights": (
                "Hugging Face weights to download and use with this model, of"
                " the format [author/repository/file]. For example,"
                " meta-llama/Llama-3.2-11B-Vision/model-00001-of-00005.safetensors"
            ),
            "version": "Llama 3.2 version.",
            "quantization_encoding": (
                "The encoding to use for a datatype that can be quantized to a"
                " low bits per weight format."
            ),
        }
