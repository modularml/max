#!/bin/bash

set -ex

# Turn off entranous output messages
export TRANSFORMERS_VERBOSITY='critical'
export TOKENIZERS_PARALLELISM='false'

CURRENT_DIR=$(dirname "$0")
MODEL_PATH="bert.torchscript"

# Example input for the model
INPUT_EXAMPLE="There are many exciting developments in the field of AI Infrastructure!"

# Download model from HuggingFace
python3 "$CURRENT_DIR/download-model.py" -o $MODEL_PATH

python3 simple-inference.py --text "$INPUT_EXAMPLE" --model-path $MODEL_PATH
