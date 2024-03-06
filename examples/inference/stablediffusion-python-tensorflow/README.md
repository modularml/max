# Stable Diffusion inference with Python

This directory illustrates how to run Stable Diffusion through MAX Engine.
Specifically, this example extracts StableDiffusion-1.4 from Keras-CV and executes
it via the MAX Engine Python API.

## Quickstart

First, install MAX as per the [MAX Engine get started
guide](https://docs.modular.com/engine/get-started/).

Then you can install the package requirements and run this example:

```bash
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
# Install the MAX Engine Python package
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
# Run the example
bash run.sh
```

## Custom Images

Getting started with your own creative prompts is as simple as:

```sh
./text-to-image.py --prompt "my image description" -o my-image.png
```

But of course, there are some additional settings that can be tweaked for more
fine-grained control over image output. See `./text-to-image.py --help` for
details.

## Files

- `download-model.py`: Downloads [keras SD
model](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion),
exports each component model as a TF `SavedModel`, compiles & loads into
MAX Engine and prints input/output info.

- `text-to-image.py`: Example program that runs full stable-diffusion pipeline
through MAX Engine in order to generate images from the given prompt.
