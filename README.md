<p align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/modular_github_logo_bg.png">
</p>

# Welcome to MAX

The Modular Accelerated Xecution ([MAX](https://www.modular.com/max)) platform is an integrated suite of AI libraries, tools, and technologies that unifies commonly fragmented AI deployment workflows. MAX accelerates time to market for the latest innovations by giving AI developers a single toolchain that unlocks full programmability, unparalleled performance, and seamless hardware portability.


<p align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/modular_architecture_diagram_bg.png">
</p>


[See here to get started with MAX](https://docs.modular.com/engine/get-started) and when you want to report issues or request features,
[please create a GitHub issue here](https://github.com/modularml/max/issues/new/choose).

The [Discord](https://discord.gg/modular) community is the best place to share your experiences and chat with the team and other community members.

In the [examples directory](https://github.com/modularml/max/tree/main/examples), you will find code examples for model inference, Jupyter notebooks for an interactive experience learning experience and instructions for how to work with benchmarking and visualization tooling.



## Getting Started

1. Install the [MAX SDK](https://docs.modular.com/engine/get-started).

2. Git clone the repository of MAX samples using the command below:

```bash
git clone https://github.com/modularml/max.git
```

## Running

### Option 1: Example console (excludes notebooks)

Run the console and follow the prompt to choose which example to run.

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
python3 examples/console.py
```

which opens up

<p align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/modular_console_bg.png">
</p>


### Option 2: Follow the README

Follow the instructions in the README for each example, notebook or tool you want to run.

### FAQ

Q: I get `ModuleNotFoundError: No module named 'max'` when running an example?

A: Please make sure you run

```sh
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
```

in your local python environment. Note that `max` can also be available in your `PATH`


## Contributing

Thanks for your interest in contributing to this repository!
We are not accepting pull requests yet.

However, we welcome your bug reports.  If you have a bug, please file an issue [here](https://github.com/modularml/max/issues/new/choose).

If you need support, the [Discord](https://discord.gg/modular) community is the best place to share your experiences and chat with the team and other community members.

## License

The Mojo examples and notebooks in this repository are licensed
under the Apache License v2.0 with LLVM Exceptions
(see the LLVM [License](https://llvm.org/LICENSE.txt)).

#### Third Party Licenses
You are entirely responsible for checking and validating the licenses of third parties (i.e. Huggingface) for related software and libraries that are downloaded.


