#!/bin/bash

# If anything goes wrong, stop running the script.
set -e

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If RoBERTa hasn't been downloaded yet, download it.
if ! [ -f ../common/roberta-pytorch/roberta.torchscript ]; then
	../common/roberta-pytorch/download-model.sh -o ../common/roberta-pytorch/roberta.torchscript
fi

# Now for the easy part -- benchmarking ;)
# PyTorch models require --input-data-schema to be specified.
max benchmark --input-data-schema=../common/roberta-pytorch/input-spec.yaml ../common/roberta-pytorch/roberta.torchscript
