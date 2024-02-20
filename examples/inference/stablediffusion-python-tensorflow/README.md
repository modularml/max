# Stable Diffusion Inference

This directory illustrates how to run Stable Diffusion through the MAX AI Engine.
Specifically, this example extracts StableDiffusion-1.4 from Keras-CV and executes
it via the MAX Python API.

## Quickstart

Once you have the MAX AI engine installed, this example can be run with:
```
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
bash run.sh
```

# Set up a Python virtual environment
python3 -m venv venv
source venv/bin/activate
# Install model-specific dependencies
python3 -m pip install -r requirements.txt
# Download the model and execute it.
./run.sh
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

- `download-model.py`

  Download [keras SD model](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion),
  export each component model as a TF `SavedModel`, compile & load into
  modular & print input/output info.

- `text-to-image.py`

  Example program which runs full stable-diffusion pipeline through the MAX AI
  Engine in order to generate images from the given prompt.
