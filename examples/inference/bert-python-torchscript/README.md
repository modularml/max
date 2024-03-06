# TorchScript BERT inference with Python

This directory includes scripts used to run simple BERT inference via the MAX
Engine Python API to predict the masked words in a sentence.

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
# Run the MAX Serving example
bash deploy.sh
```

## Scripts included

- `simple-inference.py`: Predicts the masked words in the input text using the
MAX Engine Python API. The script prepares an example input, executes the
model, and generates the filled mask.

    You can use the `--text` CLI flag to specify an input sentence.
    For example:

    ```sh
    python3 simple-inference.py --text "Paris is the [MASK] of France."
    ```

- `triton-inference.py`: Predicts the masked words in the input text using MAX
Serving. The script launches a Triton container, prepares an example input,
executes the model by calling HTTP inference endpoint, and returns the filled
mask.

    You can use the `--text` CLI flag to specify an input example.
    For example:

    ```sh
    python3 triton-inference.py --text "Paris is the [MASK] of France."
    ```
