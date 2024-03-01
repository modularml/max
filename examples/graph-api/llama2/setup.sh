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

# Usage: source setup.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE:-$0}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

export MODELS="$(pwd)/models"
export LLAMA2_DIR="$SCRIPT_DIR"
mkdir -p "$MODELS"

# Download the tokenizer table from Aydyn's llama2.mojo:

if [ ! -f "$MODELS/tokenizer.bin" ]; then
  curl https://github.com/tairov/llama2.mojo/raw/master/tokenizer.bin -L -J -o $MODELS/tokenizer.bin
fi

# Download Karpathy's TinyLlama from HF:

if [ ! -f "$MODELS/stories15M.bin" ]; then
  curl https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -L -J -o $MODELS/stories15M.bin
fi
if [ ! -f "$MODELS/stories110M.bin" ]; then
  curl https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin -L -J -o $MODELS/stories110M.bin
fi
