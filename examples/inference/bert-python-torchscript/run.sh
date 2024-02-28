#!/bin/bash

set -ex

# Example input for the model
INPUT_EXAMPLE="There are many exciting developments in the field of AI Infrastructure!"

MODEL_PATH="../../models/bert.torchscript"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 download-model.py -o "$MODEL_PATH"

python3 "$CURRENT_DIR/simple-inference.py" --text "$INPUT_EXAMPLE" --model-path "$MODEL_PATH"
