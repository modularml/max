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

#!/usr/bin/env python3

import os
from argparse import ArgumentParser

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

import tensorflow as tf
from pathlib import Path
from transformers import TFAutoModelForImageClassification

DEFAULT_MODEL_DIR = "resnet-50"
DESCRIPTION = "Download a ResNet-50 model."
HF_MODEL_NAME = "microsoft/resnet-50"


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

    print("Downloading model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, args.output_dir)
    if model_path.exists():
        print(f"Model has already been saved to {args.output_dir}/.\n")
        return

    model = TFAutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    print(f"Converting Transformers Model to Tensorflow SavedModel...")
    tf.saved_model.save(model, model_path)
    print(f"Model saved to {args.output_dir}/.\n")


if __name__ == "__main__":
    main()
