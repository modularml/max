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

export SCRIPT_PATH="$(readlink -f "${BASH_SOURCE:-$0}")"
export SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
export CUSTOM_KERNELS="$(pwd)/custom_kernels"
mkdir -p "$CUSTOM_KERNELS"

# Compile the custom RoPE kernel
mojo package "$SCRIPT_DIR/kernels/rope" -o "$CUSTOM_KERNELS/rope.mojopkg"
