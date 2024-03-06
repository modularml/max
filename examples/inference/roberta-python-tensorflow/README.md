# Tensorflow RoBERTa inference with Python

This directory includes scripts used to run simple RoBERTa inference via the
MAX Engine Python API to predict the sentiment of the given text.

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

- `download-model.py`: Downloads the model from HuggingFace, converts it to a
TensorFlow [SavedModel](https://www.tensorflow.org/guide/saved_model), and
saves it to an output directory of your choosing, or defaults to
`../../models/roberta-tensorflow/`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/microsoft/RoBERTa).

- `simple-inference.py`: Classifies example input statement using MAX
Engine. The script prepares an example input, executes the model, and generates
the resultant classification output.

    You can use the `--input` CLI flag to specify an input example.
    For example:

    ```sh
    python3 simple-inference.py --input=<YOUR_INPUT_STRING_HERE>
    ```

- `triton-inference.py`: Classifies example input image using MAX Serving.
The script launches a Triton container, prepares an example input, executes the
model by calling HTTP inference endpoint, and returns the classification
result.

    You can use the `--input` CLI flag to specify an input example.
    For example:

    ```sh
    python3 triton-inference.py --input=<YOUR_INPUT_STRING_HERE>
    ```
