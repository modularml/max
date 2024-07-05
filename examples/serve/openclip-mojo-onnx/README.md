# MAX Serve inference with ONNX OpenClip

This directory includes scripts used to run simple inference via the MAX
Serve API to classify an image.

For more information about the MAX Serve API, see the [intro to MAX
Serve](/max/serve/).

## Quickstart

1. Install MAX as per the [MAX install
guide](https://docs.modular.com/max/install/).

2. Start a virtual environment and install the package requirements:

    ```sh
    python3 -m venv venv && source venv/bin/activate
    python3 -m pip install --upgrade pip setuptools
    python3 -m pip install -r requirements.txt
    # Install the MAX Engine Python package
    python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
    ```

3. Download the ONNX OpenClip model:

    ```sh
    python3 ../../tools/common/openclip-onnx/download-model.py -o .
    ```

4. Start the server:

    ```sh
    mojo server.mojo
    ```

5. Run inference from the client:

    ```sh
    mojo client.mojo
    ```
