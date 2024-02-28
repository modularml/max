#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If ResNet50 hasn't been downloaded yet, download it.
if ! [ -d ../../models/resnet50-tensorflow ]; then
	../common/resnet50-tensorflow/download-model.sh -o ../../models/resnet50-tensorflow
fi

# Now for the easy part -- visualization ;)
max visualize ../../models/resnet50-tensorflow
if [ "$CI" != true ]; then
	python3 -m webbrowser https://netron.app || true
fi
