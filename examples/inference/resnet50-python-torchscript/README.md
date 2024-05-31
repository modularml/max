# TorchScript ResNet-50 inference with Python

This directory includes scripts used to run simple ResNet-50 inference via the
MAX Engine Python API to classify an input image. In this case, we use an image
of a leatherback turtle as an example.

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

- `download-model.py`: Downloads the model from HuggingFace, converts it to
TorchScript, and saves it to an output directory of your choosing, or defaults
to `../../models/resnet50.torchscript`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/microsoft/resnet-50).

- `simple-inference.py`: Classifies example input image using MAX Engine.
The script prepares an example input, executes the model, and generates the
resultant classification output.

    You can use the `--input` CLI flag to specify an input example.
    For example:

    ```sh
    python3 simple-inference.py --input=<path_to_input_jpg>
    ```
