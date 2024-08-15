# MAX examples

These examples demonstrate the power and flexibility of
[MAX](https://docs.modular.com/max/). They include:

## [MAX Pipelines](graph-api/pipelines/)

End-to-end demonstrations of common AI workloads and more, built using
[Mojo](https://docs.modular.com/mojo/) and the
[MAX Graph API](https://docs.modular.com/max/graph/).

## [PyTorch and ONNX inference on MAX](inference/)

MAX has the power to accelerate existing PyTorch and ONNX models directly, and
provides Python, Mojo, and C APIs for this. These examples showcase common
models from these frameworks and how to run them even faster via MAX.

## MAX Engine extensibility

We removed the extensibility API in v24.5 and are working to replace it with a
better version very soon. Because MAX is still a developer preview, we don't
want to leave any APIs in the platform that we have no intention to support. Stay
tuned for an improved extensibility API that works on CPUs and GPUs.

## [Jupyter notebooks](notebooks/)

Jupyter notebooks that showcase PyTorch and ONNX models being accelerated
through MAX.

## [Performance showcase](performance-showcase/)

A head-to-head benchmark of inference performance for common models that
demonstrates the advantages of running them with MAX.

## [Command-line tools](tools/)

MAX provides useful command-line tooling, and these examples demonstrate the
capabilities of these tools.
