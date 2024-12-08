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
"""Utilities for exploring supported pipelines."""

from max.pipelines import PIPELINE_REGISTRY, HuggingFaceFile


def list_pipelines_to_console():
    print()
    for _, arch in PIPELINE_REGISTRY.architectures.items():
        print(f"    {arch.name}")
        for version_name, version in arch.versions.items():
            print(f"        Version: {version_name}")
            for encoding_name, enc_and_strat in version.encodings.items():
                encoding, _ = enc_and_strat
                if isinstance(encoding, HuggingFaceFile):
                    print(f"            {encoding_name}: {encoding.repo_id}")
                elif isinstance(encoding, list):
                    first_encoding = encoding
                    if isinstance(first_encoding, HuggingFaceFile):
                        print(
                            f"            {encoding_name}: {first_encoding.repo_id}"
                        )
                    elif isinstance(first_encoding, list):
                        print(
                            f"            {encoding_name}: {first_encoding[0].repo_id}"
                        )
                    else:
                        print(f"            {encoding_name}: {first_encoding}")
                else:
                    print(f"            {encoding_name}: {first_encoding}")
        print()
    print()
