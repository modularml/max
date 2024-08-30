# ONNX Mistral-7b text generation with Python

This directory includes scripts to generate text with Mistral-7b via the MAX
Engine Python API.

## Quickstart

For this example, you also need a HuggingFace account and user access token
saved as the `HF_TOKEN` environment variable. To generate the access token
with READ permissions, follow the [instructions
here](https://huggingface.co/docs/hub/en/security-tokens). Copy the access
token and either set it as a permanent environment variable named `HF_TOKEN`
or use it temporarily when you run the `run.sh` script, as shown below.

Once you have your token properly set, you'll need to accept Mistral's terms
and conditions on their HuggingFace page. Please visit [this
link](https://huggingface.co/mistralai/Mistral-7B-v0.1) and accept the model's
conditions.

### Magic instructions

If you have [`magic`](https://docs.modular.com/magic), you can run the
following command:

```sh
# Run the MAX Engine example
HF_TOKEN=<your-huggingface-user-token> magic run run.sh
```

### Conda instructions

Create a Conda environment, activate that environment, and install the
requirements:

```sh
# Create a Conda environment if you don't have one
conda create -n max-repo
# Update the environment with the environment.yml file
conda env update -n max-repo -f environment.yml --prune
# Run the example
HF_TOKEN=<your-huggingface-user-token> conda run -n max-repo --live-stream ./run.sh
```

## Scripts included

- `download-model.py`: Downloads the model from HuggingFace, converts it to an
ONNX model using
[HuggingFace Optimum](https://huggingface.co/docs/optimum/index), and saves it
to an output directory of your choosing, or defaults to
`../../models/mistral7b-onnx/`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/mistralai/Mistral-7B-v0.1).

- `generate-text.py`: Generates text given a prompt using the
MAX Engine Python API. The script prepares an example input, executes the
model, and prints a generate reponse.

    You can use the `--text` CLI flag to specify an prompt.
    For example:

    ```sh
    python3 generate-text.py --text "Artificial Intelligence is"
    ```

    If no model path is specified with the `--model-path` arg, script attempts
    to load model from `../../models/mistral7b-onnx/`.
