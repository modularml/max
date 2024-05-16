#!/usr/bin/env bash
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

set -e

# Usage: ./run.sh [model_file]

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

source "$SCRIPT_DIR/setup.sh"

mojo \
    -I "$SCRIPT_DIR/tokenizer" \
    -I "$SCRIPT_DIR/.." \
    "$SCRIPT_DIR/run.🔥" \
    --model-path "${1:-$MODELS/stories15M.bin}" \
    --tokenizer-path "$MODELS/tokenizer.bin"
