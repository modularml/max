# Quantize TinyStories

**Language:** Mojo 🔥

**API**: MAX Graph

This pipeline demonstrates quantizing a full precision model using MAX Graph.
The pipeline constructs a Llama 2 model end to end in
[Mojo](https://docs.modular.com/mojo/) using the
[MAX Graph API](https://docs.modular.com/engine/graph).
In the process of graph construction, the pipeline calls the MAX Graph
[quantization](https://docs.modular.com/engine/graph/quantization) and
[checkpoint](https://docs.modular.com/engine/graph/checkpoint) APIs in order to
quantize the full precision model parameters and save them to a checkpoint.
Hence this pipeline serves as an example of how to take a full-precision model,
quantize it, and save the quantized parameters all in MAX graph.

## Model

The model is a 15M parameter model based on the Llama 2 architecture.
The original model was provided by Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) repository.
The model was trained on the
[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
which is a synthetic dataset of stories meant for training and evaluating small
LLMs with as little as 10M parameters
([arXiv paper](https://arxiv.org/abs/2305.07759)).

## Usage

1. Install MAX:

   If MAX is not already installed, follow
   [the installation instructions](https://docs.modular.com/max/install) to set
   it up on your system.

2. Run the quantize TinyStories demo:

   On first execution, this pipeline downloads the stories15M float32 model
   weights and places them in a local `.cache/` directory in your current path.
   The pipeline then loads those weights, quantizes them to a roughly 4 bit per
   weight format called `Q4_0`, and stages a graph composed of the quantized
   weights.
   After staging the full quantized Llama graph, the pipeline saves the
   resulting quantized weights as a checkpoint under the `.cache/` directory.
   On subsequent executions, the pipeline will skip the initial steps and load
   the weights from the cached quantized checkpoint.

   ```shell
   mojo ../../run_pipeline.🔥 quantize-tinystories --prompt "I believe the meaning of life is"
   ```
