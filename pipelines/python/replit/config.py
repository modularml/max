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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from max.driver import CPU, Device
from max.dtype import DType
from nn.kv_cache import KVCacheStrategy


class SupportedVersions(str, Enum):
    replit_1_5 = "1.5"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class SupportedEncodings(str, Enum):
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def dtype(self) -> DType:
        return _ENCODING_TO_DTYPE[self]

    def hf_model_name(self, version: SupportedVersions) -> str:
        if version == SupportedVersions.replit_1_5:
            return _ENCODING_TO_MODEL_NAME_REPLIT[self]
        else:
            raise ValueError(f"Unsupported version: {version}")


_ENCODING_TO_DTYPE = {
    SupportedEncodings.float32: DType.float32,
    SupportedEncodings.bfloat16: DType.bfloat16,
}


_ENCODING_TO_MODEL_NAME_REPLIT = {
    SupportedEncodings.float32: "replit-code-v1_5-3b-f32.gguf",
    SupportedEncodings.bfloat16: "replit-code-v1_5-3b-bf16.gguf",
}


@dataclass
class InferenceConfig:
    device: Device = field(default_factory=CPU)
    """Device to run inference upon."""

    weight_path: Optional[Union[str, Path]] = None
    """Path or URL of the model weights."""

    version: SupportedVersions = SupportedVersions.replit_1_5
    """Replit-code version."""

    quantization_encoding: SupportedEncodings = SupportedEncodings.float32
    """Weight encoding type."""

    seq_len: int = 4096
    """Doc me."""

    n_heads: int = 24
    """Doc me."""

    casual: bool = True
    """Doc me."""

    alibi: bool = True
    """Doc me."""

    alibi_bias_max: int = 8
    """Doc me."""

    num_blocks: int = 32
    """Doc me."""

    vocab_size: int = 32768
    """Doc me."""

    d_model: int = 3072
    """Doc me."""

    kv_n_heads: int = 8
    """Doc me."""

    max_length: int = 256

    max_new_tokens: int = 256

    serialized_model_path: Optional[str] = None

    save_to_serialized_model_path: Optional[str] = None

    n_duplicate: int = 1
    """Broadcast the static prompt `n_duplicate` times to test batching."""
    # TODO: MSDK-1095 Remove temporary `n_duplicate` cli flag.

    max_cache_batch_size: int = 16
    """Maximum cache size of sequences to the model."""

    cache_strategy: KVCacheStrategy = KVCacheStrategy.CONTINUOUS
    """Force using a specific KV cache strategy, 'naive', 'contiguous' or 'continuous'."""

    pad_to_multiple_of: int = 2
    """Pad input tensors to be a multiple of value provided."""

    def __post_init__(self) -> None:
        # Ensure quantization_encoding and kv cache strategy is consistent.
        # If a quantizated encoding is provided, we must use the naive strategy.
        if self.quantization_encoding not in [
            SupportedEncodings.float32,
            SupportedEncodings.bfloat16,
        ]:
            self.cache_strategy = KVCacheStrategy.NAIVE
