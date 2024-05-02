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

from argparse import ArgumentParser
import os

# suppress extraneous logging
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path


DEFAULT_MODEL_DIR = "../../models/minstral7b-onnx"
DESCRIPTION = "Download Minstral-7B model."
HF_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HUGGINGFACE_ERROR = """\nYou need to log into HuggingFace:
    huggingface-cli login

Then accept the terms to use Mistral:
    https://huggingface.co/mistralai/Mistral-7B-v0.1
"""


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

    print("Downloading model ...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, args.output_dir)
    if os.path.exists(model_path):
        print(f"Model has already been saved to {args.output_dir}/.\n")
        return

    print("Saving model...")
    model_path = Path(args.output_dir)
    if model_path.exists():
        print(f"'{args.output_dir}' already exists.\n")
    else:
        print("Converting the model to ONNX...")
        try:
            model = ORTModelForCausalLM.from_pretrained(
                HF_MODEL_NAME, export=True
            )
        except OSError:
            print(HUGGINGFACE_ERROR)
            exit(1)
        model.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}/.\n")


if __name__ == "__main__":
    main()
