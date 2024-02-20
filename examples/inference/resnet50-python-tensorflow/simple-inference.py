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

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from argparse import ArgumentParser
from max import engine
from PIL import Image
from transformers import AutoImageProcessor, TFAutoModelForImageClassification


DEFAULT_MODEL_DIR = "resnet-50"
DESCRIPTION = "Classify an input image."
HF_MODEL_NAME = "microsoft/resnet-50"


def execute(model_path, inputs):
    session = engine.InferenceSession()

    print("Loading and compiling model...")
    model = session.load(model_path)
    print("Model compiled.\n")

    print("Executing model...")
    outputs = model.execute(args_0=inputs["pixel_values"])
    print("Model executed.\n")
    return outputs


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--input",
        type=str,
        metavar="<jpg>",
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the downloaded model.",
    )
    args = parser.parse_args()

    # Preprocess input image
    print("Processing input...")
    image = Image.open(args.input)
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="np")
    print("Input processed.\n")

    # Classify input image
    outputs = execute(args.model_dir, inputs)

    # Extract class predictions from output
    print("Extracting class from outputs...")
    predicted_label = np.argmax(outputs["logits"], axis=-1)[0]
    model = TFAutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    predicted_class = model.config.id2label[predicted_label]

    print(
        f"\nThe input image is likely one of the following classes: \n{predicted_class}"
    )


if __name__ == "__main__":
    main()
