#!/bin/bash

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

set -ex

# Example input for the model
INPUT_EXAMPLE="There are many exciting developments in the field of AI Infrastructure!"

MODEL_PATH="../../models/bert.torchscript"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 download-model.py -o "$MODEL_PATH"

python3 "$CURRENT_DIR/simple-inference.py" --text "$INPUT_EXAMPLE" --model-path "$MODEL_PATH"
