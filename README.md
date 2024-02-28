# MAX code examples

This is a collection of sample programs, notebooks, and tools which highlight the power of the [MAX platform](https://www.modular.com/max).  This includes:

1. **inference:**
    These examples download a pre-trained model and showcase how to run it with the MAX inference engine. For certain models, deploying to Triton is also available.

2. **notebooks:**
    Includes Jupyter notebooks for an interactive learning experience.

3. **tools:**
    This demonstrates benchmarking and visualization tooling that's available to use with MAX.

## Getting Started

1. Install the [MAX SDK](https://docs.beta.modular.com/engine/get-started).

2. Git clone the repository of MAX samples using the command below:

```bash
git clone https://github.com/modularml/max.git
```

## Running

### Option 1: Example console

Run the console and follow the prompt to choose which example to run (excludes notebooks)

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
python3 examples/console.py
```

which opens up

<p align="center">
    <img src="./assets/console.png" width="500" height="450">
</p>


### Option 2: Individual example

Follow the instructions in the README for each example, notebook or tool you want to run.

### FAQ

Q: I get `ModuleNotFoundError: No module named 'max'` when running an example?

A: Please make sure you run

```sh
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
```

in your local python environment. Note that `max` can also be available in your `PATH`


## License

The Mojo examples and notebooks in this repository are licensed
under the Apache License v2.0 with LLVM Exceptions
(see the LLVM [License](https://llvm.org/LICENSE.txt)).

## Contributing

Thanks for your interest in contributing to this repository!
We are not accepting pull requests at this time.

