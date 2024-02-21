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
import numpy as np

from argparse import ArgumentParser
import tritonclient.http as httpclient
from transformers import AutoTokenizer


os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BATCH = 1
SEQLEN = 128
DEFAULT_MODEL_NAME = "bert"
DESCRIPTION = "BERT model"
HF_MODEL_NAME = "bert-base-uncased"

def execute(triton_client, model_name, inputs):
    # Set the input data
    triton_inputs = [
        httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64"),
        httpclient.InferInput(
            "attention_mask", inputs["attention_mask"].shape, "INT64"
        ),
        httpclient.InferInput(
            "token_type_ids", inputs["token_type_ids"].shape, "INT64"
        ),
    ]
    triton_inputs[0].set_data_from_numpy(inputs["input_ids"].astype(np.int64))
    triton_inputs[1].set_data_from_numpy(
        inputs["attention_mask"].astype(np.int64)
    )
    triton_inputs[2].set_data_from_numpy(
        inputs["token_type_ids"].astype(np.int64)
    )

    print("Executing model...")
    results = triton_client.infer(model_name, triton_inputs)
    print("Model executed.\n")

    return results.as_numpy("result0")


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--text",
        type=str,
        metavar="<text>",
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
        args.text,
        return_tensors="np",
        return_token_type_ids=True,
        padding="max_length",
        truncation=True,
        max_length=SEQLEN,
    )
    print("Input processed.\n")

    # Classify input statement
    outputs = execute(triton_client, args.model_name, inputs)
    
    # Extract class prediction from output
    predicted_class_id = outputs.argmax(axis=-1)[0]
    sentiment_labels = {0: "Negative", 1: "Positive"}
    print(f"Predicted sentiment: {sentiment_labels[predicted_class_id]}")


if __name__ == "__main__":
    main()
