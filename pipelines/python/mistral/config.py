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
"""All configurable parameters for Mistral."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from max.driver import DeviceSpec
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.pipelines.kv_cache import KVCacheStrategy


class SupportedVersions(str, Enum):
    mistral_nemo_instruct_2407 = "Mistral-Nemo-Instruct-2407"

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
    def quantization_encoding(self) -> Optional[QuantizationEncoding]:
        return None

    @property
    def dtype(self) -> DType:
        return _ENCODING_TO_DTYPE[self]

    def hf_model_name(self, version: SupportedVersions) -> str:
        if version == SupportedVersions.mistral_nemo_instruct_2407:
            return _ENCODING_TO_MODEL_NAME_MISTRAL[self]
        else:
            raise ValueError(f"Unsupported version: {version}")


_ENCODING_TO_DTYPE = {
    SupportedEncodings.bfloat16: DType.bfloat16,
}

_ENCODING_TO_MODEL_NAME_MISTRAL = {
    SupportedEncodings.bfloat16: "consolidated.safetensors",
}


@dataclass
class InferenceConfig:
    device_spec: DeviceSpec = DeviceSpec.cpu()
    """Device to run inference upon."""

    weight_path: Optional[Union[str, Path]] = None
    """Path or URL of the model weights."""

    huggingface_weights: Optional[
        str
    ] = "mistralai/Mistral-Nemo-Instruct-2407/consolidated.safetensors"
    """Hugging Face weights to download and use with this model."""

    version: SupportedVersions = SupportedVersions.mistral_nemo_instruct_2407
    """Mistral version."""

    # TODO: Mistral only supports "bfloat16", so this field doesn't need to be configurable by the user.
    quantization_encoding: Optional[
        SupportedEncodings
    ] = SupportedEncodings.bfloat16
    """Encoding type."""

    serialized_model_path: Optional[Union[str, Path]] = None
    """If specified, tries to load a serialized model from this path."""

    save_to_serialized_model_path: Optional[Union[str, Path]] = None
    """If specified, tries to save a serialized model to this path."""

    max_length: int = 512
    """Controls the maximum length of the text sequence (includes the input tokens)."""

    max_new_tokens: int = -1
    """Controls the maximum length of the text sequence (does not include the input tokens)."""

    n_duplicate: int = 1
    """Broadcast the static prompt `n_duplicate` times to test batching."""
    # TODO: MSDK-1095 Remove temporary `n_duplicate` cli flag.

    max_cache_batch_size: int = 16
    """Maximum cache size of sequences to the model."""

    cache_strategy: KVCacheStrategy = KVCacheStrategy.CONTINUOUS
    """Force using a specific KV cache strategy, 'naive', 'contiguous' or 'continuous'."""

    pad_to_multiple_of: int = 2
    """Pad input tensors to be a multiple of value provided."""

    max_forward_steps: int = 1
    """Maximum number of forward steps to execute."""

    repo_id: str = "mistralai/Mistral-Nemo-Instruct-2407"
    """Repo ID to use HF weights and tokenizers."""

    top_k: Optional[int] = None
    """Limits the sampling to the K most probable tokens."""

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
                " mistralai/Mistral-Nemo-Instruct-2407/consolidated.safetensors"
            ),
            "version": "Mistral version.",
            "quantization_encoding": (
                "The encoding to use for a datatype that can be quantized to a"
                " low bits per weight format."
            ),
            "max_length": (
                "Controls the maximum length of the text sequence (includes the"
                " input tokens)."
            ),
            "max_new_tokens": (
                "Controls the maximum length of the text sequence (does not"
                " include the input tokens)."
            ),
            "max_cache_batch_size": "Maximum size of sequences kept in cache.",
            "cache_strategy": (
                "Controls the batching strategy: naive, contiguous or"
                " continuous"
            ),
        }
