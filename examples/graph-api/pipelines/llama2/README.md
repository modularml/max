# Llama 2

**Language:** Mojo 🔥

**API**: MAX Graph

This pipeline demonstrates text completion from an initial prompt using the
Llama 2 large language model. The model itself has been constructed from
end to end in [the Mojo language](https://docs.modular.com/mojo/) using the
[MAX Graph API](https://docs.modular.com/engine/graph).

The MAX Graph API provides an accessible Mojo interface to the contruction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Mojo and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

The flexibility provided by MAX Graphs even includes
[the ability to define custom compute kernels](https://docs.modular.com/engine/extensibility/graph-custom-op).
An example of such a custom operation is present in this pipeline as an
optional RoPE kernel that can be loaded into the Llama 2 compute graph.

## Model

[Llama 2](https://llama.meta.com/llama2/) is an open source large language
model released by Meta. The structure of this implementation of the model was
inspired by Andrej Karpathy's [llama.c](https://github.com/karpathy/llama2.c),
and [originally written in Mojo by Aydyn Tairov](https://github.com/tairov/llama2.mojo).

The text completion demo is compatible with the the official Llama 2
[text completion demo](https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/example_text_completion.py).

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/engine/get-started)
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
   mojo ../../run_pipeline.🔥 llama2 --prompt "I believe the meaning of life is"
   ```

3. (Optional) Run with the custom RoPE kernel:

   A custom RoPE kernel has been defined in the `kernels/` directory to
   demonstrate the extensibility of the MAX graph compiler. To use that within
   this Llama 2 model, compile the Mojo package for the operation and set the
   appropriate pipeline flags using the following:

   ```shell
   source setup-custom-rope.sh && \
   mojo ../../run_pipeline.🔥 llama2 \
    --prompt "I believe the meaning of life is" \
    --custom-ops-path "$CUSTOM_KERNELS/rope.mojopkg" \
    --enable-custom-rope-kernel \
    --prompt "I believe the meaning of life is"
   ```

## Options

The following command-line options are available to customize operation of the
pipeline:

- `--batch-size`: The batch size. (Default value: `1`)
- `--model-name`: Options are `stories15M` and `stories110M`, and if not
   overridden by setting the model path, will cause weights for one of these
   pretrained models to be downloaded and used for the pipeline.
- `--model-path`: Overrides the model name, and allows for an
   already-downloaded pretrained weight file to be used with the model.
- `--custom-ops-path`: The path to a compiled Mojo package containing a custom
   graph operation to use within the pipeline.
- `--tokenizer-path`: The path to the tokenizer library to be used by the
   pipeline. (Default value: `.cache/tokenizer.bin`)
- `--enable-custom-rope-kernel`: Enables the use of the custom RoPE kernel
   within the compute graph.
- `--prompt`: The text prompt to use for further generation.
