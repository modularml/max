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

"""Download ResNet-50 TensorFlow model from HuggingFace."""

import argparse
import os
import sys
import tempfile
import traceback

# We cannot use tensorflow-cpu on Python 3.8. If we're below 3.9, we should display
# tensorflow rather than tensorflow-cpu.
MIN_TF_VERSION = (3, 9)
CUR_VERSION = sys.version_info

INSTALL_PROSE = f"""Is it installed?

If not, try:

    python3 -m venv venv; source venv/bin/activate  # if you aren't already in a virtual environment
    pip install {'tensorflow-cpu' if CUR_VERSION >= MIN_TF_VERSION else 'tensorflow'} transformers

"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        help="path to save TensorFlow SavedModel to",
        required=True,
    )
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"output {args.output!r} already exists")
        return

    print("Importing TensorFlow...", flush=True)
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="tfimport", suffix=".log", delete=False
        ) as log_file:
            traceback.print_exc(file=log_file)
        print(
            (
                f"TensorFlow module was not found.  {INSTALL_PROSE}"
                f"Detailed error info in {log_file.name}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except ImportError:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="tfimport", suffix=".log", delete=False
        ) as log_file:
            traceback.print_exc(file=log_file)
        print(
            (
                "TensorFlow installation seems to be broken.\n"
                "Make sure you can 'import tensorflow' "
                "from a Python prompt and try again.\n"
                f"Detailed error info in {log_file.name}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    tf.config.set_visible_devices([], "GPU")

    print("Importing HuggingFace transformers...", flush=True)
    try:
        from transformers import TFAutoModelForImageClassification
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
    model = TFAutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50"
    )
    print("Saving TensorFlow SavedModel...", flush=True)
    tf.saved_model.save(model, args.output)
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
