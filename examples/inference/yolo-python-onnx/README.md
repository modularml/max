# YOLOv8 Segmentation with Python

This directory illustrates how to run YOLOv8 Segmentation through the MAX Engine.
Specifically, this example uses Ultralytics YOLO with opencv to segment images
from the webcam.

## Quickstart

First, install MAX as per the [MAX Engine get started
guide](https://docs.modular.com/engine/get-started/).

Then you can install the package requirements and run this example:

```bash
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
# Install the MAX Engine Python package
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
# Run the example
bash run.sh
```

## Performance Comparison

To get an idea of the performance difference between the ONNX Runtime and MAX Engine,
you can run the script with/without the `--onnx` flag.

The model execution time is printed as part of the window title.
Note, it can take a bit for the execution time to stabilize.

MAX Engine:

```sh
./segment-webcam.py
```

ONNX Runtime:

```sh
./segment-webcam.py --onnx
```

## Files

- `download-model.py`: Downloads YOLOv8n-seg from
[ultralytics](https://github.com/ultralytics/ultralytics)
and exports it as ONNX.

- `segment-webcam.py`: Example program that runs the full YOLO segmentation pipeline
through the MAX Engine on images from the webcam.
