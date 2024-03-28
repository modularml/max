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
from PIL import Image
from transformers import AutoImageProcessor, TFAutoModelForImageClassification
import tritonclient.http as httpclient

DEFAULT_MODEL_NAME = "resnet-50"
DESCRIPTION = "Classify an input image."
HF_MODEL_NAME = "microsoft/resnet-50"


def execute(triton_client, model_name, inputs):
    # Set the input data
    triton_inputs = [
        httpclient.InferInput("args_0", inputs["pixel_values"].shape, "FP32")
    ]
    triton_inputs[0].set_data_from_numpy(inputs["pixel_values"])

    print("Executing model...")
    results = triton_client.infer(model_name, triton_inputs).as_numpy("logits")
    print("Model executed.\n")

    return results


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
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name to execute inference.",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    args = parser.parse_args()

    # Create a triton client
    triton_client = httpclient.InferenceServerClient(url=args.url)

    # Preprocess input image
    print("Processing input...")
    image = Image.open(args.input)
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="np")
    print("Input processed.\n")

    # Classify input image
    outputs = execute(triton_client, args.model_name, inputs)

    # Extract class predictions from output
    print("Extracting class from outputs...")
    predicted_label = np.argmax(outputs, axis=-1)[0]
    model = TFAutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    predicted_class = model.config.id2label[predicted_label]

    print(
        "\nThe input image is likely one of the following classes:"
        f" \n{predicted_class}"
    )


if __name__ == "__main__":
    main()
