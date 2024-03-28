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
# Wrapper around download-model.py that sets up a temporary venv & installs
# dependencies if needed.

set -e

script="$(dirname "$0")/download-model.py"

# Check to see if PyTorch and HuggingFace Transformers are installed.
if python3 -c 'import torch; import transformers' >& /dev/null; then
	python3 "$script" "$@"
else
	# Required dependencies not installed -- set up a venv and install them
	# in there, then run the script (cleaning up the venv afterwards).
	venv_dir="$(mktemp -d)"
	python3 -m venv "$venv_dir"
	"$venv_dir/bin/pip" install -U setuptools pip wheel
	"$venv_dir/bin/pip" install torch transformers
	"$venv_dir/bin/python3" "$script" "$@"
	rm -rf "$venv_dir"
fi
