# Tensorflow BERT inference with C

This directory includes scripts used to run simple BERT inference via the MAX
Engine C API to predict the masked text. In this case, we use the text _"The
capital of France is [MASK]."_ as an example.

## Quickstart

First, install MAX as per the [MAX Engine get started
guide](https://docs.modular.com/engine/get-started/).

Then you can install the package requirements and run this example:

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
bash run.sh
```

## Scripts included

- `download-model.py`: Downloads the model from HuggingFace, converts it to a
`TensorFlow SavedModel`, and saves it to directory `bert`.

    For more information about the model, reference the [model
    card](https://huggingface.co/bert-base-uncased).

- `generate-inputs.py`: Prepares an example input and saves the pre-processed
input to a local directory. Example:

  ```sh
  python3 generate-inputs.py --input="Paris is the [MASK] of France."
  ```

- `post-process.py`: Loads the generated output, post-processes it, and outputs
the prediction. Example:

  ```sh
  python3 post-process.py --input="Paris is the [MASK] of France."
  ```

## Building the example

This example uses CMake. To build the executable, please use the following
commands:

```sh
export MAX_PKG_DIR=`modular config max.path`
cmake -B build -S .
cmake --build build
```

The executable is called `bert` and will be present in the build directory.

## Usage

- Make sure `bin` directory of `max` package is in `PATH`.

```sh
./build/bert ../../models/bert-tensorflow
```
