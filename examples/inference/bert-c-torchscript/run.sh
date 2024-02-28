#!/bin/bash

set -e

# set MAX path
MAX_PKG_DIR="$(modular config max.path)"
export MAX_PKG_DIR

CURRENT_DIR=$(dirname "$0")
MODEL_PATH="$CURRENT_DIR/../../models/bert.torchscript"

# Example input for the model
INPUT_EXAMPLE="My dog is cute."

# Download model from HuggingFace
python3 "$CURRENT_DIR/download-model.py" --text "$INPUT_EXAMPLE" -o "$MODEL_PATH"

# Build the example
cmake -B build -S "$CURRENT_DIR"
cmake --build build

# Run example
./build/bert "$MODEL_PATH"

# Post process
python3 "$CURRENT_DIR/post-process.py"
