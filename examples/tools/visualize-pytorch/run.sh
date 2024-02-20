#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If ResNet50 hasn't been downloaded yet, download it.
if ! [ -f ../common/resnet50-pytorch/resnet50.torchscript ]; then
	../common/resnet50-pytorch/download-model.sh -o ../common/resnet50-pytorch/resnet50.torchscript
fi

# Now for the easy part -- visualization ;)
max visualize --input-data-schema=../common/resnet50-pytorch/input-spec.yaml ../common/resnet50-pytorch/resnet50.torchscript
python3 -m webbrowser https://netron.app || true
