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

from .llama3_token_gen import Llama3TokenGenerator
from max.pipelines import (
    HuggingFaceFile,
    SupportedArchitecture,
    SupportedVersion,
    SupportedEncoding,
)
from max.pipelines.kv_cache import KVCacheStrategy
from nn.tokenizer import TextTokenizer

llama_arch = SupportedArchitecture(
    name="llama",
    versions=[
        SupportedVersion(
            name="3",
            encodings={
                SupportedEncoding.float32: (
                    HuggingFaceFile("modularai/llama-3", "llama-3-8b-f32.gguf"),
                    [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.bfloat16: (
                    HuggingFaceFile(
                        "modularai/llama-3",
                        "llama-3-8b-instruct-bf16.gguf",
                    ),
                    [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.q4_0: (
                    HuggingFaceFile(
                        "modularai/llama-3",
                        "llama-3-8b-instruct-q4_0.gguf",
                    ),
                    [KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.q4_k: (
                    HuggingFaceFile(
                        "modularai/llama-3",
                        "llama-3-8b-instruct-q4_k_m.gguf",
                    ),
                    [KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.q6_k: (
                    HuggingFaceFile(
                        "modularai/llama-3",
                        "llama-3-8b-instruct-q6_k.gguf",
                    ),
                    [KVCacheStrategy.NAIVE],
                ),
            },
            default_encoding=SupportedEncoding.float32,
        ),
        SupportedVersion(
            name="3.1",
            encodings={
                SupportedEncoding.float32: (
                    HuggingFaceFile(
                        "modularai/llama-3.1",
                        "llama-3.1-8b-instruct-f32.gguf",
                    ),
                    [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.bfloat16: (
                    HuggingFaceFile(
                        "modularai/llama-3.1",
                        "llama-3.1-8b-instruct-bf16.gguf",
                    ),
                    [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.q4_0: (
                    HuggingFaceFile(
                        "modularai/llama-3.1",
                        "llama-3.1-8b-instruct-q4_0.gguf",
                    ),
                    [KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.q4_k: (
                    HuggingFaceFile(
                        "modularai/llama-3.1",
                        "llama-3.1-8b-instruct-q4_k_m.gguf",
                    ),
                    [KVCacheStrategy.NAIVE],
                ),
                SupportedEncoding.q6_k: (
                    HuggingFaceFile(
                        "modularai/llama-3.1",
                        "llama-3.1-8b-instruct-q6_k.gguf",
                    ),
                    [KVCacheStrategy.NAIVE],
                ),
            },
            default_encoding=SupportedEncoding.float32,
        ),
    ],
    default_version="3.1",
    token_generator=Llama3TokenGenerator,
    tokenizer=TextTokenizer,
)
