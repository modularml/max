# Llama 3.1

**Language:** Mojo 🔥

**API**: MAX Graph

This pipeline demonstrates text completion from an initial prompt using the
Llama 3.1 large language model. The model itself has been constructed from
end to end in [the Mojo language](https://docs.modular.com/mojo/) using the
[MAX Graph API](https://docs.modular.com/engine/graph).

The MAX Graph API provides an accessible Mojo interface to the construction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Mojo and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

> [!WARNING]
> The current Mojo interface for the MAX Graph API is deprecated, in favor of
> the new Python-based Graph API. We recommend exploring
> [the Python versions of these pipelines](../../../../pipelines/python/),
> which include new models and capabilities.

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

The easiest way to try out this pipeline is with our Magic command-line tool.
[Follow the instructions to install Magic](https://docs.modular.com/magic).
Once installed, you can try out text generation using Llama 3.1 with the
following command:

```sh
magic run llama3 --prompt "I believe the meaning of life is"
```

On first execution, the tokenizer library and model weights will be
downloaded and placed in a `.cache/modular` subdirectory within your home
directory. The model will then be compiled and text completion will begin from
the specified prompt.

To modify or build upon the pipeline code, you can use the following steps:

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
   cd max/examples/graph-api/pipelines/llama3/
   ```

3. Run the text completion demo:

   All of the pipelines have been configured to use a common driver, located
   in the directory hosting all MAX Graph examples. Assuming you're starting
   at the path of this README, the command invocation will look like:

   ```shell
   mojo ../../run_pipeline.🔥 llama3 --prompt "I believe the meaning of life is"
   ```

## Options

The following command-line options are available to customize operation of the
pipeline:

- `--model-path`: Overrides the default URL, and allows for an
  already-downloaded pretrained weight file to be used with the model.
- `--custom-ops-path`: The path to a compiled Mojo package containing a custom
   graph operation to use within the pipeline.
- `--tokenizer-path`: The path to the tokenizer library to be used by the
   pipeline. (Default value: `.cache/tokenizer.bin`)
- `--max-length`: The context length of the model.
  (Default value: 512)
- `--max-new-tokens`: The maximum number of new tokens to generate. If a -1
  value is provided, the model will continue to generate tokens for the entire
  context length. (Default value: -1)
- `--min-p`: The starting required percentage for
  [Min P sampling](https://github.com/ggerganov/llama.cpp/pull/3841).
  (Default value: 0.05)
- `--prompt`: The text prompt to use for further generation.
- `--quantization-encoding`: The encoding to use for a datatype that can be
  quantized to a low bits per weight format. The options for quantized formats
  will download and cache default weights, but `float32` requires the use of
  `--model-path` to specify locally downloaded full-precision weights for use
  in the model.
  Valid values: `q4_0`, `q4_k`, `q6_k`, `float32`.
  (Default value: `q4_k`).
- `--temperature`: The temperature for sampling with 0.0 being greedy sampling.
  (Default value: 1.0)
- `--version`: Selects which version in the Llama 3 family to use.
  Valid values: `3.0`, `3.1`.
  (Default value: `3.1`)
- `--warmup-pipeline`: Performs a warmup run of the pipeline before text
  generation.
