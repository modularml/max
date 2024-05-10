#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Download OpenCLIP ONNX model from HuggingFace."""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

INSTALL_PROSE = """Is it installed?

If not, try:

    python3 -m venv venv; source venv/bin/activate  # if you aren't already in a virtual environment
    python3 -m pip install torch

"""


def log_fatal(prefix, msg_fn):
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=prefix, suffix=".log", delete=False
    ) as log_file:
        traceback.print_exc(file=log_file)
        print(msg_fn(log_file), file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output_dir",
        help="directory to save ONNX model to",
        required=True,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_onnx_path = output_dir / "clip.onnx"
    model_out_path = output_dir / "clip-dynamic-quint8.onnx"
    if os.path.exists(model_onnx_path):
        print(f"output {model_onnx_path!r} already exists")
        return
    if os.path.exists(model_out_path):
        print(f"output {model_out_path!r} already exists")
        return

    print("Importing PyTorch...", flush=True)
    try:
        import torch
    except ModuleNotFoundError:
        log_fatal(
            "torchimport",
            lambda log_file: f"Torch module was not found.  {INSTALL_PROSE}Detailed error info in {log_file.name}",
        )
    except ImportError:
        log_fatal(
            "torchimport",
            lambda log_file: f"PyTorch installation seems to be broken.\nMake sure you can 'import torch' from a Python prompt and try again.\nDetailed error info in {log_file.name}",
        )

    print("Importing ONNX Runtime...", flush=True)
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ModuleNotFoundError:
        log_fatal(
            "onnxruntime",
            lambda log_file: f"ONNX Runtime module was not found.  {INSTALL_PROSE}Detailed error info in {log_file.name}",
        )
    except ImportError:
        log_fatal(
            "onnxruntime",
            lambda log_file: f"ONNX Runtime installation seems to be broken.\nMake sure you can 'import onnxruntime' from a Python prompt and try again.\nDetailed error info in {log_file.name}",
        )

    print("Importing OpenCLIP...", flush=True)
    try:
        import open_clip
    except ModuleNotFoundError:
        log_fatal(
            "openclip",
            lambda log_file: f"OpenCLIP module was not found.  {INSTALL_PROSE}Detailed error info in {log_file.name}",
        )
    except ImportError:
        log_fatal(
            "openclip",
            lambda log_file: f"OpenCLIP installation seems to be broken.\nMake sure you can 'import open_clip' from a Python prompt and try again.\nDetailed error info in {log_file.name}",
        )

    print("Getting pre-trained model...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print("Exporting ONNX...", flush=True)
    image_size = model.visual.image_size
    image = torch.randn(10, 3, image_size[0], image_size[1])
    text = tokenizer(["cat", "dog", "fish"])
    text = text.int()
    with open(model_onnx_path, "wb") as out_fstream:
        torch.onnx.export(
            model,
            (image, text),
            model_onnx_path,
            dynamic_axes={"image": {0: "batch"}},
            verbose=True,
            input_names=["image", "text"],
            output_names=["image_features", "text_features", "logit_scale"],
        )
    print("Quantizing ONNX...", flush=True)
    quantize_dynamic(
        model_onnx_path,
        model_out_path,
        per_channel=False,
        weight_type=QuantType.QUInt8,
        extra_options={"WeightSymmetric": False},
    )
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
