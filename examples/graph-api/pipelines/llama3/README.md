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
`q4_0` quantized encodings.

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/engine/get-started)
   to set it up on your system.

2. Install Python dependencies.

   This enables using the HuggingFace
   [transformers](https://github.com/huggingface/transformers) AutoTokenizer.
   If `transformers` isn't found, a Mojo tokenizer implementation is used.

   ```shell
   python3 -m pip install -r requirements.txt
   ```

3. Run the text completion demo:

   **To access the llama models, you need to agree to their license in Huggingface.**

   License is located here [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

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

- `--batch-size`: The batch size. (Default value: `1`)
- `--model-path`: Overrides the default URL, and allows for an
  already-downloaded pretrained weight file to be used with the model.
- `--custom-ops-path`: The path to a compiled Mojo package containing a custom
   graph operation to use within the pipeline.
- `--tokenizer-path`: The path to the tokenizer library to be used by the
   pipeline. (Default value: `.cache/tokenizer.bin`)
- `--prompt`: The text prompt to use for further generation.
- `--quantization-encoding`: The encoding to use for a datatype that can be
  quantized to a low bits per weight format.
  Valid values: `q4_0`, `float32`.
  (Default value: `q4_0`).
