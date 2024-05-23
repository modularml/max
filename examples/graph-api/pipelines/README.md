# MAX Pipelines

These are end-to-end pipelines that demonstrate the power of
[MAX](https://docs.modular.com/max/) for accelerating common AI workloads, and
more. The umbrella `pipelines` [Mojo](https://docs.modular.com/mojo/) module
contains these pipelines as their own modules, along with shared modules
hosting common functionality.

## Pipelines

The pipelines include:

- [Llama 2](llama2): A text completion demo using the Llama 2 model, implemented
in Mojo using the MAX Graph API.
- [Replit Code](replit): Code generation via the Replit Code V1.5 3B mode,
implemented in Mojo using the MAX Graph API.

Instructions for how to run each pipeline can be found in their respective
subdirectories. A shared `run_pipeline.🔥` Mojo driver is used to execute
the pipelines.

## Shared modules

In addition to the pipelines, common modules contain types and functions shared
between the various pipelines. These modules currently include:

- [weights](weights): A module containing code for loading common weight
formats, such as
[GGUF](https://github.com/ggerganov/ggml/blob/cce2ac9a5d788c3b6bb72a3b3dbde9247d8b85a7/docs/gguf.md).
