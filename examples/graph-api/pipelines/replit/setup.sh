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

# Usage: source setup.sh

export SCRIPT_PATH="$(readlink -f "${BASH_SOURCE:-$0}")"
export SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
export LOCAL_PATH_REPLIT=$(pwd)/.cache/replit
mkdir -p $LOCAL_PATH_REPLIT

DTYPE="float32"
CONVERTED_PATH=$LOCAL_PATH_REPLIT/converted_float32

if [[ -z $1 ]]; then
    :  # Use default
elif [[ $1 == "float32" ]]; then
    :  # Already set as default.
elif [[ $1 == "bfloat16" ]]; then
    DTYPE="bfloat16"
    CONVERTED_PATH=$LOCAL_PATH_REPLIT/converted_bfloat16
else
    echo "Unsupported type: $1. Only float32 and bfloat16 are supported."
    return 127
fi

if [ ! -f "$LOCAL_PATH_REPLIT/pytorch_model.bin" ]; then
    curl https://huggingface.co/replit/replit-code-v1_5-3b/resolve/main/pytorch_model.bin -L -J -o $LOCAL_PATH_REPLIT/pytorch_model.bin
fi

if [ -z "$(ls -A $CONVERTED_PATH)" ]; then
    echo Converting $LOCAL_PATH_REPLIT/pytorch_model.bin...
    python3 -m pip install -r $SCRIPT_DIR/requirements.txt
    python3 $SCRIPT_DIR/weights/convert_pytorch_ckpt.py \
        --input $LOCAL_PATH_REPLIT/pytorch_model.bin \
        --output_dir $CONVERTED_PATH --output_type $DTYPE
    echo "Checkpoint converted and written out to $CONVERTED_PATH (dtype=$DTYPE)."
else
    echo "Checkpoint already converted and written out to $CONVERTED_PATH (dtype=$DTYPE)."
fi
