# Tensorflow RoBERTa Inference

This directory includes scripts used to run simple RoBERTa inference via the
MAX Engine and Mojo APIs to predict the sentiment of the given text.

## Quickstart

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
# Install the MAX Engine Python package
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
# Run the MAX Engine example
bash run.sh
```

## Scripts Included

- `download-model.py`
    Downloads the model from HuggingFace, converts it to a TensorFlow
    [SavedModel](https://www.tensorflow.org/guide/saved_model),
    and saves it to an output directory of your choosing, or defaults to
    `../../models/roberta-tensorflow/`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/microsoft/RoBERTa).

- `simple-inference.py`
    Classifies example input statement using the MAX Engine. The script prepares an
    example input, executes the model, and generates the resultant classification
    output.

    You can use the `--input` CLI flag to specify an input example.
    For example, `python3 simple-inference.py --input=<YOUR_INPUT_STRING_HERE>`.

