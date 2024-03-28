# TorchScript BERT inference with C

This directory includes scripts used to run simple BERT inference via the MAX
Engine C API to predict the sentiment of the given text.

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

- `pre-process.py`: Prepares an example input and saves the pre-processed input
to a local directory, for use use in the `main.c` program. Example:

    ```sh
    python3 pre-process.py --text "Paris is the [MASK] of France."
    ```

- `post-process.py`: Loads the generated output, post-processes it, and outputs
the prediction. Example:

    ```sh
    python3 post-process.py
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
python3 ../common/bert-torchscript/download-model.py
python3 pre-process.py --text "Your text"
./build/bert ../../models/bert.torchscript
python3 post-process.py
```
