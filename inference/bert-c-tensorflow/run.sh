#!/bin/bash

set -ex

# Turn off entranous output messages
export TF_CPP_MIN_LOG_LEVEL='3'
export TRANSFORMERS_VERBOSITY='critical'
export TOKENIZERS_PARALLELISM='false'

CURRENT_DIR=$(dirname "$0")
MODEL_DIR="$CURRENT_DIR/bert-tf-model"

# Download model from HuggingFace
python3 "$CURRENT_DIR/download-model.py" "--output-dir=$MODEL_DIR"

# Execute the model with example input
INPUT_EXAMPLE="The capital of France is [MASK]."

# Generate inputs for inference
python3 "$CURRENT_DIR/generate-inputs.py" --input "$INPUT_EXAMPLE"

# Build the example
cmake -B build -S "$CURRENT_DIR"
cmake --build build

# Run example
./build/bert "$MODEL_DIR"

# Post process and cleanup
python3 "$CURRENT_DIR/post-process.py" --input "$INPUT_EXAMPLE"
