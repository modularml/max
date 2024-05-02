# MAX Engine Tools

MAX comes bundled with a super-handy `max` CLI tool that lets you perform a
couple of common tasks on many ML models without writing a single line of code.

We currently provide two tools:

- `benchmark`: Benchmark models with MLPerf load generation.
- `visualize`: Show a model graph as interpreted by MAX Engine in Netron.

Try one out with `benchmark-pytorch/run.sh` or `visualize-pytorch/run.sh` to
benchmark or visualize a PyTorch model.

## What's going on?

Either of these scripts start by downloading a ResNet50 model (a
model typically used for image classification) and converting it into the
proper format -- for PyTorch, this is [TorchScript].  This model will be stored
in `../models/resnet50.torchscript`.

TorchScript does not include any metadata for shapes the model expects as
input, so we need to provide an `--input-data-schema`.  For this ResNet-50
model, we've already written one for you in
`common/resnet50-pytorch/input-spec.yaml`, which looks like this:

```yaml
inputs:
  - input_name: pixel_values
    shape: 1x3x224x224xf32
    compile_shape: ?x3x224x224xf32
```

Benchmarking and visualization works as follows:

```sh
% max benchmark --input-data-schema=common/resnet50-pytorch/input-spec.yaml ../models/resnet50.torchscript
% # Benchmarking results will be printed by above command
% max visualize --input-data-schema=common/resnet50-pytorch/input-spec.yaml ../models/resnet50.torchscript
% # Above command will generate a file that can be opened by netron.app
```

Learn more about `max benchmark`, `max visualize`, and input data schemas in
the [MAX documentation].

  [TorchScript]: https://pytorch.org/docs/stable/jit.html
  [MAX documentation]: https://docs.modular.com/engine/
