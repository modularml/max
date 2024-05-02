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

# If anything goes wrong, stop running the script.
set -e
MODEL_PATH="../../models/bert-mlm.torchscript"
MODEL_DIR="bert-mlm"
INPUT_EXAMPLE="Paris is the [MASK] of France."


# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 ../common/bert-torchscript/download-model.py -o "$MODEL_PATH" --mlm

echo "Preparing model repository"
# Triton expects models to reside in the specific layout, i.e.
# <model-repository-path>/
#   <model-name>/
#     [config.pbtxt]
#     <version>/
#       <model-definition-file>
# see https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
# https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#tensorflow-models

mkdir -p model-repository/${MODEL_DIR}/1/
cp -r ${MODEL_PATH} model-repository/${MODEL_DIR}/1/
cp config.pbtxt model-repository/${MODEL_DIR}/

echo "Starting container"
CONTAINER_ID=$(\
  docker run --rm -d --net=host \
    -v $(pwd)/model-repository:/model-repository \
    public.ecr.aws/modular/max-serving \
    tritonserver --model-repository=/model-repository \
    --load-model=${MODEL_DIR} \
)

printf "Compiling ${MODEL_DIR} model"
until curl --output /dev/null --silent --fail localhost:8000/v2/health/ready; do
    printf '.'
    sleep 5
done
printf "\nMAX Serving container started\n\n"


# Execute the model with example input
python3 triton-inference.py --input "$INPUT_EXAMPLE"

printf "\nStopping container..."
docker stop $CONTAINER_ID > /dev/null
printf "\nDone.\n"
