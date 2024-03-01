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

"""Download RoBERTa PyTorch model from HuggingFace."""

import argparse
import os
import sys
import tempfile
import traceback
import urllib.request


INSTALL_PROSE = """Is it installed?

If not, try:

    python3 -m venv venv; source venv/bin/activate  # if you aren't already in a virtual environment
    pip install torch

"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        help="path to save TorchScript model to",
        required=True,
    )
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"output {args.output!r} already exists")
        return

    print("Importing PyTorch...", flush=True)
    try:
        import torch
    except ModuleNotFoundError:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="torchimport", suffix=".log", delete=False
        ) as log_file:
            traceback.print_exc(file=log_file)
        print(
            (
                f"Torch module was not found.  {INSTALL_PROSE}"
                f"Detailed error info in {log_file.name}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except ImportError:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="torchimport", suffix=".log", delete=False
        ) as log_file:
            traceback.print_exc(file=log_file)
        print(
            (
                "PyTorch installation seems to be broken.\n"
                "Make sure you can 'import torch' "
                "from a Python prompt and try again.\n"
                f"Detailed error info in {log_file.name}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print("Importing HuggingFace transformers...", flush=True)
    try:
        from transformers import RobertaForSequenceClassification
    except ModuleNotFoundError:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="hfimport", suffix=".log", delete=False
        ) as log_file:
            traceback.print_exc(file=log_file)
        print(
            (
                "HuggingFace transformers module was not found. "
                f" {INSTALL_PROSE}Detailed error info in {log_file.name}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except ImportError:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="hfimport", suffix=".log", delete=False
        ) as log_file:
            traceback.print_exc(file=log_file)
        print(
            (
                "HuggingFace transformers installation seems to be broken.\n"
                "Make sure you can 'import transformers' "
                "from a Python prompt and try again.\n"
                f"Detailed error info in {log_file.name}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print("Getting pre-trained model...", flush=True)
    model = RobertaForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
    )
    print("Tracing model...", flush=True)
    traced_model = torch.jit.trace(
        model,
        [
            torch.zeros(1, 128, dtype=torch.int64),
            torch.ones(1, 128, dtype=torch.float32),
            torch.zeros(1, 128, dtype=torch.int64),
        ],
        strict=False,
    )
    print("Saving TorchScript...", flush=True)
    traced_model.save(args.output)
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
