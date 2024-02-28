#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

MODEL_DIR="../../models/stable-diffusion-tensorflow"
NPROMPT="ugly, bad anatomy, weird tongue"
PPROMPT="Cute puppy chewing on a big steak"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model
python3 ../common/stable-diffusion-tensorflow/download-model.py -o "$MODEL_DIR"

# Execute model
mojo text-to-image.🔥 --seed 4 --num-steps 15 \
     --prompt "$PPROMPT" --negative-prompt "$NPROMPT" --model-dir "$MODEL_DIR"
