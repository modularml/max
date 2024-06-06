# Llama 2

**Language:** Mojo 🔥

**API**: MAX Graph

This pipeline demonstrates text completion from an initial prompt using the
Llama 2 large language model. The model itself has been constructed from
end to end in [the Mojo language](https://docs.modular.com/mojo/) using the
[MAX Graph API](https://docs.modular.com/max/graph).

The MAX Graph API provides an accessible Mojo interface to the contruction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Mojo and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

The flexibility provided by MAX Graphs even includes
[the ability to define custom compute kernels](https://docs.modular.com/max/extensibility/graph-custom-op).
An example of such a custom operation is present in this pipeline as an
optional RoPE kernel that can be loaded into the Llama 2 compute graph.

## Model

[Llama 2](https://llama.meta.com/llama2/) is an open source large language
model released by Meta. The structure of this implementation of the model was
inspired by Andrej Karpathy's [llama.c](https://github.com/karpathy/llama2.c),
and [originally written in Mojo by Aydyn Tairov](https://github.com/tairov/llama2.mojo).

The text completion demo is compatible with the the official Llama 2
[text completion demo](https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/example_text_completion.py).

The default settings for this pipeline use the 7B set of pretrained weights in
`q4_0` quantized encodings.

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/max/install)
   to set it up on your system.

2. (Optional) Install Python dependencies.

   This enables using the HuggingFace
   [transformers](https://github.com/huggingface/transformers) AutoTokenizer.
   If `transformers` isn't found, a Mojo tokenizer implementation is used.

   ```shell
   python3 -m pip install -r requirements.txt
   ```

3. Run the text completion demo:

   **To access the llama models, you need to agree to their license in Huggingface.**

   License is located here [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)

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

4. (Optional) Run with the custom RoPE kernel:

   A custom RoPE kernel has been defined in the `kernels/` directory to
   demonstrate the extensibility of the MAX graph compiler. To use that within
   this Llama 2 model, compile the Mojo package for the operation and set the
   appropriate pipeline flags using the following:

   ```shell
   source setup-custom-rope.sh && \
   mojo ../../run_pipeline.🔥 llama2 \
    --prompt "I believe the meaning of life is" \
    --custom-ops-path "$CUSTOM_KERNELS/rope.mojopkg" \
    --enable-custom-rope-kernel
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
- `--enable-custom-rope-kernel`: Enables the use of the custom RoPE kernel
   within the compute graph.
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
  (Default value: `q4_0`).
- `--temperature`: The temperature for sampling, on a scale from 0.0 - 1.0,
  with 0.0 being greedy sampling. (Default value: 0.5)

## Ideas for future extension

There are many ways that this pipeline can be built upon or extended, and
this is a short list of suggestions for future work:

- Enhance the tokenizer so that it can stand alone as a general-purpose
tokenizer for multiple text generation pipelines.
- Expand the customizable options for text generation.
- Incorporate and use weights from other models.
- Improve the quality of the text generation.
- Identify performance bottlenecks and further tune time-to-first-token and
throughput.
