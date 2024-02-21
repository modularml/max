#!/bin/bash
# Wrapper around download-model.py that sets up a temporary venv & installs
# dependencies if needed.

set -e

script="$(dirname "$0")/download-model.py"

# Check to see if TensorFlow & HuggingFace Transformers are installed.
if python3 -c 'import tensorflow; import transformers' >& /dev/null; then
	python3 "$script" "$@"
else
	# Required dependencies not installed -- set up a venv and install them
	# in there, then run the script (cleaning up the venv afterwards).
	venv_dir="$(mktemp -d)"
	python3 -m venv "$venv_dir"
	"$venv_dir/bin/pip" install -U setuptools pip wheel
	"$venv_dir/bin/pip" install tensorflow transformers
	"$venv_dir/bin/python3" "$script" "$@"
	rm -rf "$venv_dir"
fi
