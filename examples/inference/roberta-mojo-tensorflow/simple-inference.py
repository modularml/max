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

from max import engine

import os

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from argparse import ArgumentParser
from transformers import AutoTokenizer, TFRobertaForSequenceClassification

DEFAULT_MODEL_DIR = "../../models/roberta-tensorflow"
DESCRIPTION = "Identify the sentiment of an input statement."
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"


def execute(model_path, inputs):
    session = engine.InferenceSession()

    print("Loading and compiling model...")
    model = session.load(model_path)
    print("Model compiled.\n")

    print("Executing model...")
    outputs = model.execute(**inputs)
    print("Model executed.\n")
    return outputs


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
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the downloaded model.",
    )
    args = parser.parse_args()

    # Preprocess input statement
    print("Processing input...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    inputs = tokenizer(
        args.input, return_tensors="np", return_token_type_ids=True
    )
    print("Input processed.\n")

    # Classify input statement
    outputs = execute(args.model_dir, inputs)

    # Extract class prediction from output
    print("Extracting class from outputs...")
    predicted_class_id = outputs["logits"].argmax(axis=-1)[0]
    model = TFRobertaForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    classification = model.config.id2label[predicted_class_id]

    print(f"The sentiment of the input statement is: {classification}")


if __name__ == "__main__":
    main()
