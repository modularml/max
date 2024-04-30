For Demo 1:

    - To package the custom ops, run `mojo package custom_ops`.
    - Run `mojo max_graph.mojo` and compare the accuracy of the different ops.

For Demo 2:

    - Export the onnx model `$MODULAR_PYTHON onnx_det.py`
    - Attempt to benchmark the onnx model, get an error: 
    ```bash
        max benchmark onnx_det.onnx
    ```
    - To visualize the with the custom op, run
    ```bash
        max visualize onnx_det.onnx --custom_ops_path=custom_ops.mojopkg 
    ```
    - Benchmark the model with
    ```bash
    MOJO_PYTHON_LIBRARY=$(modular config mojo.python_lib) \
    max benchmark onnx_det.onnx --custom-ops-path custom_ops.mojopkg
    ```
