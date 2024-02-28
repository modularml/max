#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

MODEL_DIR="../../models/resnet50-tensorflow/"
INPUT_EXAMPLE="input/leatherback_turtle.jpg"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 download-model.py -o "$MODEL_DIR"

# Execute the model with example input
python3 simple-inference.py --input "$INPUT_EXAMPLE" --model-dir "$MODEL_DIR"
