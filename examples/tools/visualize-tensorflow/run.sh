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

# If ResNet50 hasn't been downloaded yet, download it.
if ! [ -d ../../models/resnet50-tensorflow ]; then
	../common/resnet50-tensorflow/download-model.sh -o ../../models/resnet50-tensorflow
fi

# Now for the easy part -- visualization ;)
max visualize ../../models/resnet50-tensorflow
if [ "$CI" != true ]; then
	python3 -m webbrowser https://netron.app || true
fi
