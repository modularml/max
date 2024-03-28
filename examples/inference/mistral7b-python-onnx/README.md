# ONNX Mistral-7b text generation with Python

This directory includes scripts to generate text with Mistral-7b via the MAX
Engine Python API.

## Quickstart

First, install MAX as per the [MAX Engine get started
guide](https://docs.modular.com/engine/get-started/).

Then you can install the package requirements and run this example:

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
# Install the MAX Engine Python package
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
# Run the MAX Engine example
bash run.sh
```

## Scripts included

- `download-model.py`: Downloads the model from HuggingFace, converts it to an
ONNX model using
[HuggingFace Optimum](https://huggingface.co/docs/optimum/index), and saves it
to an output directory of your choosing, or defaults to
`../../models/mistral7b-onnx/`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/mistralai/Mistral-7B-v0.1).

- `generate-text.py`: Generates text given a prompt using the
MAX Engine Python API. The script prepares an example input, executes the
model, and prints a generate reponse.

    You can use the `--text` CLI flag to specify an prompt.
    For example:

    ```sh
    python3 generate-text.py --text "Artificial Intelligence is"
    ```

    If no model path is specified with the `--model-path` arg, script attempts
    to load model from `../../models/mistral7b-onnx/`.
