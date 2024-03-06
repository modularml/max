# Stable Diffusion inference with Mojo

This directory illustrates how to run Stable Diffusion through MAX Engine.
Specifically, this example extracts StableDiffusion-1.4 from Keras-CV and
executes it via the MAX Engine Mojo API.

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

## Custom images

Getting started with your own creative prompts is as simple as:

```sh
mojo text-to-image.🔥 --prompt "my image description" -o my-image.png
```

To refine images there are a few additional options:

- `--seed <int>`: Control PRNG initialization (default: 0)
- `--num-steps <int>`: Set # of denoising iterations (default: 25)
- `--negative-prompt <str>`: Textual description of items or styles to avoid.
  (default: None)
