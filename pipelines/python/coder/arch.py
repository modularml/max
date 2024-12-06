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

from max.pipelines import (
    HuggingFaceFile,
    SupportedArchitecture,
    SupportedEncoding,
    SupportedVersion,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .model import CoderModel

coder_arch = SupportedArchitecture(
    name="DeepseekCoder",
    versions=[
        SupportedVersion(
            name="1.5",
            encodings={
                SupportedEncoding.bfloat16: (
                    [
                        HuggingFaceFile(
                            "deepseek-ai/deepseek-coder-7b-instruct-v1.5", f
                        )
                        for f in [
                            "model-00001-of-00003.safetensors",
                            "model-00002-of-00003.safetensors",
                            "model-00003-of-00003.safetensors",
                        ]
                    ],
                    [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.NAIVE],
                )
            },
            default_encoding=SupportedEncoding.bfloat16,
        ),
    ],
    default_version="1.5",
    pipeline_model=CoderModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
