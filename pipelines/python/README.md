# Llama 3.1

**Language:** Python

**API**: MAX Graph

This pipeline demonstrates text completion from an initial prompt using the
Llama 3.1 large language model. The model itself has been constructed in Python
using the [MAX Graph API](https://docs.modular.com/engine/graph).

The MAX Graph API provides an accessible interface to the construction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Python and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

## Model

[Llama 3.1](https://llama.meta.com/llama3/) is an open source large language
model released by Meta. The structure of this implementation was inspired by
Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and its [Mojo
port by Aydyn Tairov](https://github.com/tairov/llama2.mojo).

The text completion demo is compatible with the the official Llama 3
[text completion demo](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/example_text_completion.py).

The default settings for this pipeline use the 8B set of pretrained weights in
`q4_k` quantized encodings.

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/max/install)
   to set it up on your system.

2. Clone the MAX examples repository:

   If you don't already have a local clone of this repository, create one via:

   ```shell
   git clone https://github.com/modularml/max.git
   ```

   The following instructions assume that you're present within this pipeline's
   directory, and you can change to it after cloning:

   ```shell
   cd max/pipelines/python/
   ```

3. Run the text completion demo:

   On first execution, the tokenizer library and model weights will be
   downloaded and placed in a local `.cache/` directory in your current path.
   The model will then be compiled and text completion will begin from the
   specified prompt.

   All of the pipelines have been configured to use a common driver, located
   in the directory hosting all MAX Graph examples. Assuming you're starting
   at the path of this README, the command invocation will look like:

   ```shell
   python3 pipelines.py llama3 --prompt "I believe the meaning of life is"
   ```

4. Optionally host a text completion endpoint via MAX Serve.

   MAX Serve provides functionality to host performant OpenAI compatible
   endpoints using the FastAPI framework.

   You can configure the pipeline to be hosted by using the `--serve` argument.
   For e.g.,

   ```shell
   python3 pipelines.py llama3 --quantization-encoding q4_k --serve
   ```

   A request can be submitted via a cURL command.

   ```shell
   curl -N http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3",
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
    ```

## Options

The following command-line options are available to customize operation of the
pipeline:

- `--max-length`: Controls the maximum length of the text sequence
  (includes the input tokens).
  (Default value: 512)
- `--max-new-tokens`: The maximum number of new tokens to generate. If a -1
  value is provided, the model will continue to generate tokens for the entire
  context length. (Default value: -1)
  `--model-path` to specify locally downloaded full-precision weights for use
  in the model.
  Valid values: `q4_0`, `q4_k`, `q6_k`, `float32`.
  (Default value: `float32`).
- `--prompt`: The text prompt to use for further generation.
- `--quantization-encoding`: The encoding to use for a datatype that can be
  quantized to a low bits per weight format. The options for quantized formats
  will download and cache default weights, but `float32` requires the use of
- `--serialized-model-path`: If specified, tries to load a serialized model
  from this path.
- `--version`: Selects which version in the Llama 3 family to use.
  Valid values: `3`, `3.1`.
  (Default value: `3.1`)
- `--weight-path`: Overrides the default URL, and allows for an
  already-downloaded pretrained weight file to be used with the model.