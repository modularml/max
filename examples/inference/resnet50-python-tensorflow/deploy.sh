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

# If anything goes wrong, stop running the script.
set -e
MODEL_DIR="resnet-50"
INPUT_EXAMPLE="input/leatherback_turtle.jpg"


# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 download-model.py -o "$MODEL_DIR"

echo "Preparing model repository"
# Triton expects models to reside in the specific layout, i.e.
# <model-repository-path>/
#   <model-name>/
#     [config.pbtxt]
#     <version>/
#       <model-definition-file>
# see https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
# https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#tensorflow-models

mkdir -p model-repository/resnet-50/1/model.savedmodel/
cp -r resnet-50/* model-repository/resnet-50/1/model.savedmodel/
cp config.pbtxt model-repository/resnet-50/

echo "Starting container"
CONTAINER_ID=$(\
  docker run --rm -d --net=host --ipc=host \
    -v $(pwd)/model-repository:/model-repository \
    public.ecr.aws/modular/max-serving-de \
    tritonserver --model-repository=/model-repository \
    --load-model=resnet-50 \
)

printf "Compiling ${MODEL_DIR} model"
until curl --output /dev/null --silent --fail localhost:8000/v2/health/ready; do
    printf '.'
    sleep 5
done
printf "\nMAX Serving container started\n"

# Execute the model with example input
python3 triton-inference.py --input "$INPUT_EXAMPLE"

printf "\nStopping container..."
docker stop $CONTAINER_ID > /dev/null
printf "\nDone.\n"
