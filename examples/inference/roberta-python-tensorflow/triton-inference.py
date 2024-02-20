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
import tritonclient.http as httpclient
from transformers import AutoTokenizer, TFRobertaForSequenceClassification

DEFAULT_MODEL_NAME = "roberta"
DESCRIPTION = "Identify the sentiment of an input statement."
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"


def execute(triton_client, model_name, inputs):
    # Set the input data
    triton_inputs = [
        httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT32"),
        httpclient.InferInput(
            "attention_mask", inputs["attention_mask"].shape, "INT32"
        ),
        httpclient.InferInput(
            "token_type_ids", inputs["token_type_ids"].shape, "INT32"
        ),
    ]
    triton_inputs[0].set_data_from_numpy(inputs["input_ids"].astype(np.int32))
    triton_inputs[1].set_data_from_numpy(
        inputs["attention_mask"].astype(np.int32)
    )
    triton_inputs[2].set_data_from_numpy(
        inputs["token_type_ids"].astype(np.int32)
    )

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
        metavar="str",
        required=True,
        help="Statement to classify.",
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

    # Preprocess input statement
    print("Processing input...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    inputs = tokenizer(
        args.input, return_tensors="np", return_token_type_ids=True
    )
    print("Input processed.\n")

    # Classify input statement
    outputs = execute(triton_client, args.model_name, inputs)

    # Extract class predictions from output
    print("Processing outputs...")
    predicted_class_id = outputs.argmax(axis=-1)[0]
    model = TFRobertaForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    classification = model.config.id2label[predicted_class_id]
    print(f"The sentiment of the input statement is: {classification}")


if __name__ == "__main__":
    main()
