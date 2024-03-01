# TorchScript BERT Inference

This directory includes scripts used to run simple BERT inference via the MAX Engine C API to predict the sentiment of the given text.

## Quickstart

```
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
bash run.sh
```

## Scripts Included

- `pre-process.py`
    Loads the generated output, post-processes it, and outputs the prediction.
    Usage: `python3 pre-process.py --text <masked text>`

- `post-process.py`
    Loads the generated output, post-processes it, and outputs the prediction.
    Usage: `python3 post-process.py`.

## Building the example

This example uses CMake. To build the executable, please use the following commands:

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
