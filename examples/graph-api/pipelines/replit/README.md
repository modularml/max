# Replit Code V1.5 3B

**Language:** Mojo 🔥

**API**: MAX Graph

This pipeline demonstrates code completion from an initial prompt using
Replit's Code V1.5 3B large language model. The model itself has been
constructed from end to end in
[the Mojo language](https://docs.modular.com/mojo/) using the
[MAX Graph API](https://docs.modular.com/max/graph).

The MAX Graph API provides an accessible Mojo interface to the contruction of
flexible accelerated compute graphs, which are then optimized by the MAX
Engine's advanced graph compiler. This pipeline showcases how a large language
model can be fully defined using Mojo and MAX Graphs and then compiled for
optimal inference performance via the MAX Engine.

## Model

[Replit Code](https://blog.replit.com/replit-code-v1_5) is an open source code
generation model trained on permissively licensed code and released by
[Replit](https://replit.com). The V1.5, 3B variant is the basis for this
implementation, and weights are
[obtained via Hugging Face](https://huggingface.co/replit/replit-code-v1-3b).

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/max/install)
   to set it up on your system.

2. Download and convert the model weights:

   Before the first execution of the pipeline, weights need to be downloaded
   and converted into the correct format for use by this model. This weight
   conversion process requires the use of PyTorch, which currently is only
   compatible with Python 3.11 or older on macOS. PyTorch and all
   dependencies will be automatically installed, and weights will be
   downloaded and converted by running the following script:

   ```shell
   source setup.sh
   ```

3. Run the code completion demo:

   Invoking the pipeline will cause the model graph to be compiled and code
   generation will begin from the specified prompt.

   All of the pipelines have been configured to use a common driver, located
   in the directory hosting all MAX Graph examples. Assuming you're starting
   at the path of this README, the command invocation will look like:

   ```shell
   mojo ../../run_pipeline.🔥 replit --prompt 'def hello():\n  print("hello world")'
   ```

## Options

The following command-line options are available to customize operation of the
pipeline:

- `--converted-weights-path`: Specifies the path to the converted model
   weights. (Default value: `.cache/replit/converted`)
- `--prompt`: The text prompt to use for further code generation.

## Ideas for future extension

This isn't an exhaustive list, but here are some ideas for ways in which this
pipeline may be extended or improved:

- Replace the SentencePiece tokenizer with one written in Mojo. Currently,
the tokenizer is loaded from the `transformers` library via Python
interoperability and it might be useful to have this all in Mojo.
- Incorporate 4-bit quantization.
- Improve the quality of the code generation.
- Identify performance bottlenecks and further tune time-to-first-token and
throughput.
