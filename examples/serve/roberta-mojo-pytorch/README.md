# MAX Serve inference with Roberta TorchScript

This directory includes scripts used to run simple inference via the MAX
Serve API to predict the sentiment of the given text.

For more information about the MAX Serve API, see the [intro to MAX
Serve](/max/serve/).

## Quickstart

1. Install MAX as per the [MAX install
guide](https://docs.modular.com/max/install/).

2. Start a virtual environment and install the package requirements:

    ```sh
    python3 -m venv venv && source venv/bin/activate

    python3 -m pip install --upgrade pip setuptools
    ```

    MAX Python package dependency:

    ```sh
    python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
    ```

    Dependencies for this example:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

3. Download the TorchScript model:

    ```sh
    python3 download-roberta.py
    ```

4. Start the server:

    ```sh
    mojo server.mojo
    ```

5. Run inference from the client:

    ```sh
    mojo client.mojo
    ```

You should see results like this:

```text
The sentiment is: joy
```
