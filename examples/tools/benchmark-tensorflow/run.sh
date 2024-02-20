#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If ResNet50 hasn't been downloaded yet, download it.
if ! [ -d ../common/resnet50-tensorflow/resnet50-savedmodel ]; then
	../common/resnet50-tensorflow/download-model.sh -o ../common/resnet50-tensorflow/resnet50-savedmodel
fi

# Now for the easy part -- benchmarking ;)
max benchmark ../common/resnet50-tensorflow/resnet50-savedmodel
