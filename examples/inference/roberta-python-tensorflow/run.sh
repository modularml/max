#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

INPUT_EXAMPLE="There are many exciting developments in the field of AI Infrastructure!"
MODEL_DIR="roberta"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model from HuggingFace
python3 download-model.py -o "$MODEL_DIR"

# Execute the model with example input
python3 simple-inference.py --input "$INPUT_EXAMPLE" --model-dir "$MODEL_DIR"
