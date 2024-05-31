# TorchScript BERT inference with Mojo

This directory includes scripts used to run simple BERT inference via the [MAX
Engine Mojo API](https://docs.modular.com/max/reference/mojo/engine/) to
predict the masked words in a sentence.

## Quickstart

1. Install MAX as per the [MAX install
    guide](https://docs.modular.com/max/install/).

2. Install the package requirements:

    ```sh
    python3 -m venv venv && source venv/bin/activate
    python3 -m pip install --upgrade pip setuptools
    python3 -m pip install -r requirements.txt
    # Install the MAX Engine Python package
    python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
    ```

3. Run the example with the Bash script, which downloads the BERT TorchScript
   model and runs it with the input `"Paris is the [MASK] of France."`:

    ```sh
    bash run.sh
    ```
