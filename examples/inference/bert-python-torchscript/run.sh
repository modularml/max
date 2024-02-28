#!/bin/bash

set -ex

CURRENT_DIR=$(dirname "$0")
MODEL_PATH="$CURRENT_DIR/../../models/bert.torchscript"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 download-model.py -o "$MODEL_PATH"

python3 "$CURRENT_DIR/simple-inference.py" --text "$INPUT_EXAMPLE" --model-path "$MODEL_PATH"
