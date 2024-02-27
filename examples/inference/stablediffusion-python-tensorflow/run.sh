#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

MODEL_DIR="stable-diffusion"
NPROMPT="bad anatomy, looking away, looking sideways, crooked stick"
NPROMPT="$NPROMPT, stick not going through jaw, orange tongue"
PPROMPT="Cute puppy chewing on a stick"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# Download model
python3 ../common/stable-diffusion/download-model.py -o "$MODEL_DIR"

# Execute model
python3 text-to-image.py --seed=7 --num-steps=20 --prompt "$PPROMPT" --negative "$NPROMPT" --model-dir "$MODEL_DIR"
