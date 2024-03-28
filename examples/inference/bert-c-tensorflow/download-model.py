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
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from transformers import TFBertForMaskedLM

HF_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_DIR = "../../models/bert-tensorflow"


def main():
    parser = ArgumentParser(description="Download model for inference.")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Location to save the model",
        default=DEFAULT_MODEL_DIR,
    )

    print("Downloading model ...")
    args = parser.parse_args()

    tf.config.set_visible_devices([], "GPU")

    model_path = Path(args.output_dir).resolve()
    if os.path.exists(model_path):
        print(f"Model has already been saved to {model_path}/.\n")
        return

    model = TFBertForMaskedLM.from_pretrained(HF_MODEL_NAME)
    print("Converting Transformers Model to Tensorflow SavedModel...")
    tf.saved_model.save(model, model_path)
    print(f"Model saved to {model_path}.\n")


if __name__ == "__main__":
    main()
