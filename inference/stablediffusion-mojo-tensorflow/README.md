# Stable Diffusion Inference

This directory illustrates how to run Stable Diffusion through the MAX AI Engine.
Specifically, this example extracts StableDiffusion-1.4 from Keras-CV and executes
it via the MAX Mojo API.

## Quickstart

Once you have the MAX AI engine installed, this example can be run with:

```
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
bash run.sh
```

## Custom Images

Getting started with your own creative prompts is as simple as:

```
mojo text-to-image.🔥 --prompt "my image description" -o my-image.png
```

To refine images there are a few additional options:

  - `--seed <int>`: Control PRNG initialization (default: 0)
  - `--num-steps <int>`: Set # of denoising iterations (default: 25)
  - `--negative-prompt <str>`: Textual description of items or styles to avoid. (default: None)
  
## Tips & Tricks

These models can take quite a long time to compile. The `MODULAR_MAX_ENABLE_MODEL_IR_CACHE`
environment variable controls whether or not the compiled binaries are cached within the
MAX engine. Be sure that it's set to `true` so the models are only compiled once.
