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

import torch
from transformers import AutoTokenizer

from max import engine

BATCH = 1
SEQLEN = 128
DEFAULT_MODEL_PATH = "bert.torchscript"
DESCRIPTION = "BERT model"
HF_MODEL_NAME = "bert-base-uncased"


def execute(model_path, text, input_dict):
    session = engine.InferenceSession()

    input_spec_list = [
        engine.TorchInputSpec(shape=tensor.size(), dtype=engine.DType.int64)
        for tensor in input_dict.values()
    ]
    options = engine.TorchLoadOptions(input_spec_list)

    model = session.load(model_path, options)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQLEN,
    )

    outputs = model.execute(**inputs)

    print("Loading and compiling model...")
    model = session.load(model_path, options)
    print("Model compiled.\n")

    print("Executing model...")
    outputs = model.execute(**inputs)
    print("Model executed.\n")
    return outputs


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
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Directory for the downloaded model.",
    )
    args = parser.parse_args()

    input_dict = {
        "input_ids": torch.zeros((BATCH, SEQLEN), dtype=torch.int64),
        "attention_mask": torch.zeros((BATCH, SEQLEN), dtype=torch.int64),
        "token_type_ids": torch.zeros((BATCH, SEQLEN), dtype=torch.int64),
    }

    outputs = execute(args.model_path, args.text, input_dict)

    # Extract class prediction from output
    predicted_class_id = outputs["result0"]["logits"].argmax(axis=-1)[0]
    predicted_label = predicted_class_id.item()
    sentiment_labels = {0: "Positive", 1: "Negative"}
    print(f"Predicted sentiment: {sentiment_labels[predicted_label]}")


if __name__ == "__main__":
    main()
