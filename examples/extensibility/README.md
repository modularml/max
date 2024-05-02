# MAX extensibility examples

Regardless of the model format you have (such as PyTorch, ONNX, or MAX Graph),
MAX allows you to write custom ops in Mojo that the MAX Engine compiler can
natively analyze and optimize along with with the rest of the model.

For more information and API walkthroughs, see the [MAX extensibility
documentation](https://docs.modular.com/engine/extensibility/).

This directory includes two examples that show you how to add custom ops to
either an ONNX model or a MAX Graph model.

## Add a custom op to an ONNX model

1. Run the `onnx-model.py` script to generate the `onnx-model.onnx` model:

   ```sh
   python3 -m pip install -r requirements.txt
   ```

   ```sh
   python3 onnx-model.py
   ```

2. The `onnx-model.onnx` model you get currently does not compile with MAX Engine
   because it includes the DET op that's currently not implemented in MAX.
   As proven if you try to benchmark it:

   ```sh
   max benchmark onnx-model.onnx
   ```

3. Package the `det.mojo` custom op in `custom_ops` with this command:

   ```sh
   mojo package custom_ops
   ```

4. Now run the model with the custom op:

   ```sh
   export MOJO_PYTHON_LIBRARY=$(modular config mojo-max.python_lib)
   ```

   ```sh
   max benchmark onnx-model.onnx --custom-ops-path=custom_ops.mojopkg
   ```

   ```sh
   python3 onnx-inference.py
   ```

For more details, see [how to create a custom
op](https://docs.modular.com/engine/extensibility/custom-op).

## Add a custom op to a MAX Graph model

1. Package the `gelu.mojo` custom op in `custom_ops` with this command (if
   you didn't already from above):

   ```sh
   mojo package custom_ops
   ```

2. Build and run a graph using the GELU custom op:

   ```sh
   mojo max-graph.mojo
   ```

For more details, see [how to create a custom op in MAX
Graph](https://docs.modular.com/engine/extensibility/graph-custom-op).
