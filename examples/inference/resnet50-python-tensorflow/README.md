# Tensorflow ResNet-50 Inference

This directory includes scripts used to run simple Resnet-50 inference via the
MAX Engine to classify an input image. In this case, we use an image of a
leatherback turtle as an example.

## Quickstart

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

## Scripts Included

- `download-model.py`
    Downloads the model from HuggingFace, converts it to a TensorFlow
    [SavedModel](https://www.tensorflow.org/guide/saved_model),
    and saves it to an output directory of your choosing, or defaults
    to `../../models/resnet50-tensorflow/`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/microsoft/resnet-50).

- `simple-inference.py`
    Classifies example input image using the MAX Engine. The script prepares an
    example input, executes the model, and generates the resultant classification
    output.

    You can use the `--input` CLI flag to specify an input example.
    For example, `python3 simple-inference.py --input=<path_to_input_jpg>`.

- `triton-inference.py`
    Classifies example input image using the MAX Serving container. The script launches a Triton container, prepares an example input, executes the model by calling HTTP inference endpoint, and returns the classification result.

    To specify a input example, use `python3 triton-inference.py --input=<path_to_input_jpg>`.
