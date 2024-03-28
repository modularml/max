#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

# If anything goes wrong, stop running the script.
set -e

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If RoBERTa hasn't been downloaded yet, download it.
if ! [ -f ../../models/roberta.torchscript ]; then
	../common/roberta-pytorch/download-model.sh -o ../../models/roberta.torchscript
fi

# Now for the easy part -- benchmarking ;)
# PyTorch models require --input-data-schema to be specified.
max benchmark --input-data-schema=../common/roberta-pytorch/input-spec.yaml ../../models/roberta.torchscript
