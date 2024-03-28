#!/bin/bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##

set -ex

# Example input for the model
INPUT_EXAMPLE="Paris is the [MASK] of France."

MODEL_PATH="../../models/bert-mlm.torchscript"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 ../common/bert-torchscript/download-model.py -o "$MODEL_PATH" --mlm

python3 simple-inference.py --text "$INPUT_EXAMPLE" --model-path "$MODEL_PATH"
