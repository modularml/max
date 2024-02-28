# MAX Engine Tools

MAX comes bundled with a super-handy `max` CLI tool that lets you perform a
couple of common tasks on many ML models without writing a single line of code.

We currently provide two tools:

- `benchmark`: Benchmark models with MLPerf load generation.
- `visualize`: Show a model graph as interpreted by MAX Engine in Netron.

Try one out with `benchmark-tensorflow/run.sh` or `visualize-tensorflow/run.sh`
to benchmark or visualize a TensorFlow model, or `benchmark-pytorch/run.sh` or
`visualize-pytorch/run.sh` to benchmark or visualize a PyTorch model.

## What's going on?

Either of these scripts start by downloading a ResNet50 model (a
model typically used for image classification) and converting it into the
proper format -- for TensorFlow, this is a [SavedModel], and for PyTorch, this
is [TorchScript].  This model will be stored in
`../models/resnet50-tensorflow` or
`../models/resnet50.torchscript`.  Then, we run `max benchmark`
or `max visualize` on this model.  TensorFlow needs no extra options:

```sh
% max benchmark ../models/resnet50-tensorflow
% # Benchmarking results will be printed by above command
% max visualize ../models/resnet50-tensorflow
% # Above command will generate a file that can be opened by netron.app
```

PyTorch requires one extra step.  TorchScript does not include any metadata for
shapes the model expects as input, so we need to provide an
`--input-data-schema`.  For this ResNet-50 model, we've already written one for
you in `common/resnet50-pytorch/input-spec.yaml`, which looks like this:

```yaml
inputs:
  - input_name: pixel_values
    shape: 1x3x224x224xf32
    compile_shape: ?x3x224x224xf32
```

Benchmarking and visualization works similarly, just with an extra option:

```sh
% max benchmark --input-data-schema=common/resnet50-pytorch/input-spec.yaml ../models/resnet50.torchscript
% # Benchmarking results will be printed by above command
% max visualize --input-data-schema=common/resnet50-pytorch/input-spec.yaml ../models/resnet50.torchscript
% # Above command will generate a file that can be opened by netron.app
```

`--input-data-schema` can be provided for TensorFlow as well if desired, though
it is not usually required unless the model takes a dynamic input shape or you
want to override the way input data is generated for benchmarking.

Learn more about `max benchmark`, `max visualize`, and input data schemas in
the [MAX documentation].

  [MAX documentation]: https://docs.modular.com/engine/
