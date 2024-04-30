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

import os
import signal
import subprocess
from argparse import ArgumentParser


DEFAULT_MODEL_DIR = "../../models/stable-diffusion-onnx"
DESCRIPTION = "Download a Stable Diffusion model."


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Output directory for the downloaded model.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Download and export model from huggingface.
    if os.path.exists(args.output_dir):
        print(f"Skipping {args.output_dir} (already exists)")
        return

    print("Downloading Model and Exporting...\n")
    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            "runwayml/stable-diffusion-v1-5",
            args.output_dir,
        ]
    ).check_returncode()
    print(f"Model exported to {args.output_dir}/.\n")


if __name__ == "__main__":
    main()
