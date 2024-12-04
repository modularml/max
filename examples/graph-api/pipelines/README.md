# MAX Pipelines

These are end-to-end pipelines that demonstrate the power of
[MAX](https://docs.modular.com/max/) for accelerating common AI workloads, and
more. The umbrella `pipelines` [Mojo](https://docs.modular.com/mojo/) module
contains these pipelines as their own modules, along with shared modules
hosting common functionality.

> [!WARNING]
> The current Mojo interface for the MAX Graph API is deprecated, in favor of
> the new Python-based Graph API. We recommend exploring
> [the Python versions of these pipelines](../../../pipelines/python/), which
> include new models and capabilities.

## Pipelines

The pipelines include:

- [Llama 3](llama3): A text completion demo using the Llama 3 model,
implemented in Mojo using the MAX Graph API. This pipeline contains everything
needed to run a self-hosted large language model.
- [Llama 2](llama2): Similar to the Llama 3 text generation pipeline, only
with the Llama 2 model. The Llama 2 pipeline also shows how to use a custom
kernel in MAX Graphs.
- [Replit Code](replit): Code generation via the Replit Code V1.5 3B mode,
implemented in Mojo using the MAX Graph API.
- [Quantize TinyStories](quantize_tinystories): A demonstration of quantizing
a full-precision model using the MAX Graph API, originally trained on the
TinyStories dataset.

Instructions for how to run each pipeline can be found in their respective
subdirectories. A shared `run_pipeline.🔥` Mojo driver is used to execute
the pipelines.

## Shared modules

In addition to the pipelines, common modules contain types and functions shared
between the various pipelines. These modules currently include:

- [nn](./nn/): Abstractions for common layers in neural network architectures.
- [tokenizer](./tokenizer/): Shared tokenizers used across text pipelines.
- [weights](./weights/): A module containing code for loading common weight
formats, such as
[GGUF](https://github.com/ggerganov/ggml/blob/cce2ac9a5d788c3b6bb72a3b3dbde9247d8b85a7/docs/gguf.md).

## Tests

Unit tests for the pipelines and their shared components have been provided in
a directory parallel to this one. They can be run using the Mojo testing
framework via an invocation similar to the following (if running from this
directory):

```shell
mojo test ../ -I ../
```

To select a single test case to run, an invocation like the following can be
used:

```shell
mojo test -I ../ "../test/llama3/test_heap.mojo::test_simple()"
```
