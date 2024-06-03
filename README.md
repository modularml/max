<p align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/modular_github_logo_bg.png">
</p>

# Welcome to MAX

The Modular Accelerated Xecution ([MAX](https://www.modular.com/max)) platform
is an integrated suite of AI libraries, tools, and technologies that unifies
commonly fragmented AI deployment workflows. MAX accelerates time to market
for the latest innovations by giving AI developers a single toolchain that
unlocks full programmability, unparalleled performance, and seamless hardware portability.

<p align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/modular_architecture_diagram_bg.png">
</p>

[See here to get started with MAX](https://docs.modular.com/engine/get-started)
and when you want to report issues or request features,
[please create a GitHub issue here](https://github.com/modularml/max/issues/new/choose).

The [Discord](https://discord.gg/modular) community is the best place to share
your experiences and chat with the team and other community members.

In the [examples directory](https://github.com/modularml/max/tree/main/examples),
you will find code examples for model inference, Jupyter notebooks for an
interactive experience learning experience and instructions for how to work
with benchmarking and visualization tooling.

## Getting Started

MAX is available in both stable and nightly builds. To install either version,
follow the guide to [install the MAX SDK](https://modul.ar/get-started) (also
see for the system requirements).

Then clone this repository:

```bash
git clone https://github.com/modularml/max.git
```

If you installed the nightly build, be sure you switch to the `nightly` branch,
because the `main` branch is for stable releases and might not be compatible
with nightly builds:

```bash
git checkout nightly
```

## Running

### MAX Pipelines

To show off the full power of MAX, a
[series of end-to-end pipelines for common AI workloads](./examples/graph-api/pipelines/)
(and more) are ready to run. As one example, this includes everything needed to
self-host
[the Llama 3 text-generation model](./examples/graph-api/pipelines/llama3/).
These pipelines are completely written in Mojo, and all code is provided so
that they can be customized, built upon, or learned from.

### Examples

In addition to the end-to-end pipelines, there are many [examples](./examples/)
that exercise various aspects of MAX, from
[performing inference using PyTorch and ONNX models](./examples/inference/) to
[demonstrating command-line tooling capabilities](./examples/tools/).

You can follow the instructions in the README for each example,
notebook or tool you want to run.

### Notebooks

Check out the [notebooks examples](./examples/notebooks/) for using MAX Engine
🏎️ for models such as

- [Mistral-7B](./examples/notebooks/mistral7b-python-onnx.ipynb)
- [Roberta-pytorch](./examples/notebooks/roberta-python-pytorch.ipynb)

### FAQ

Q: I get `ModuleNotFoundError: No module named 'max'` when running an example?

A: Please make sure you run

```sh
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
```

in your local python environment. Note that `max` can also be available in your `PATH`

### MAX Serving Docker Container

For MAX Serving, you can pull our Docker Container from the the public ECR here:
[https://gallery.ecr.aws/modular/max-serving](https://gallery.ecr.aws/modular/max-serving)

```public.ecr.aws/modular/max-serving```

## Contributing

Thanks for your interest in contributing to this repository!
We are not accepting pull requests yet.

However, we welcome your bug reports.  If you have a bug, please file an issue
[here](https://github.com/modularml/max/issues/new/choose).

If you need support, the [Discord](https://discord.gg/modular)
community is the best place to share your experiences and chat with
the team and other community members.

## License

The Mojo examples and notebooks in this repository are licensed
under the Apache License v2.0 with LLVM Exceptions
(see the LLVM [License](https://llvm.org/LICENSE.txt)).

### Third Party Licenses

You are entirely responsible for checking and validating the licenses of
third parties (i.e. Huggingface) for related software and libraries that are downloaded.
