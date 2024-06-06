# Llama 3

**Language:** Mojo 🔥

**API**: MAX Graph

This pipeline demonstrates text completion from an initial prompt using the
Llama 3 large language model. The model itself has been constructed from
end to end in [the Mojo language](https://docs.modular.com/mojo/) using the
[MAX Graph API](https://docs.modular.com/engine/graph).

The MAX Graph API provides an accessible Mojo interface to the construction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Mojo and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

## Model

[Llama 3](https://llama.meta.com/llama3/) is an open source large language
model released by Meta. The structure of this implementation of the model was
inspired by Andrej Karpathy's [llama.c](https://github.com/karpathy/llama2.c),
and [originally written in Mojo by Aydyn
Tairov](https://github.com/tairov/llama2.mojo), which were originally written
to run Llama 2.

The text completion demo is compatible with the the official Llama 3
[text completion demo](https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/example_text_completion.py).

The default settings for this pipeline use the 8B set of pretrained weights in
`q4_k` quantized encodings.

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/max/install)
   to set it up on your system.

2. Run the text completion demo:

   On first execution, the tokenizer library and model weights will be
   downloaded and placed in a local `.cache/` directory in your current path.
   The model will then be compiled and text completion will begin from the
   specified prompt.

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
- `--max-tokens`: The maximum number of tokens to generate.
  (Default value: 512)
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
- `--temperature`: The temperature for sampling, on a scale from 0.0 - 1.0,
  with 0.0 being greedy sampling. (Default value: 0.5)
