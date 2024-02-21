#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If RoBERTa hasn't been downloaded yet, download it.
if ! [ -d ../common/roberta-tensorflow/roberta-savedmodel ]; then
	../common/roberta-tensorflow/download-model.sh -o ../common/roberta-tensorflow/roberta-savedmodel
fi

# Now for the easy part -- benchmarking ;)
# Even though this is a TensorFlow model, we need --input-data-schema in order to override how the inputs are generated, since the inputs must have a fixed range and the model has dynamic input shapes.
max benchmark  --input-data-schema=../common/roberta-tensorflow/input-spec.yaml ../common/roberta-tensorflow/roberta-savedmodel
