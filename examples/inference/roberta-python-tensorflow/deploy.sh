#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

INPUT_EXAMPLE="There are many exciting developments in the field of AI Infrastructure!"
MODEL_DIR="../../models/roberta-tensorflow"

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

mkdir -p model-repository/${MODEL_DIR}/1/model.savedmodel/
cp -r ${MODEL_DIR}/* model-repository/${MODEL_DIR}/1/model.savedmodel/
cp config.pbtxt model-repository/${MODEL_DIR}/

echo "Starting container"
CONTAINER_ID=$(\
  docker run --rm -d --net=host --ipc=host \
    -v $(pwd)/model-repository:/model-repository \
    public.ecr.aws/o4d0h4e7/max-serving-de \
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
