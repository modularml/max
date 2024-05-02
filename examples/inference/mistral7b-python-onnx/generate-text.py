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


from max import engine

import os

# suppress extraneous logging
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

import signal
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList


DEFAULT_MODEL_PATH = "../../models/minstral7b-onnx/model.onnx"
DESCRIPTION = "Generate text given a prompt."
HF_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HUGGINGFACE_ERROR = """\nYou need to log into HuggingFace:
    huggingface-cli login

Then accept the terms to use Mistral:
    https://huggingface.co/mistralai/Mistral-7B-v0.1
"""

# Number of tokens to generate
N_TOKENS = 8


def generate(model_path, input_ids):
    session = engine.InferenceSession()
    maxmodel = session.load(f"{model_path}/model.onnx")

    print("\nGenerating text...")

    # Values used in generation loop
    inputs = {}
    N_BATCH = 1
    N_LAYERS = 32
    N_HEADS = 8
    KV_LEN = 128

    # Initialize the additional layer to 0 for the first iteration:
    for i in range(N_LAYERS):
        inputs[f"past_key_values.{i}.key"] = torch.zeros(
            [N_BATCH, N_HEADS, 0, KV_LEN], dtype=torch.float
        ).numpy()
        inputs[f"past_key_values.{i}.value"] = torch.zeros(
            [N_BATCH, N_HEADS, 0, KV_LEN], dtype=torch.float
        ).numpy()

    current_seq = input_ids

    logits_processor = LogitsProcessorList()
    for idx in range(N_TOKENS):
        # Prepare inputs dictionary
        inputs["input_ids"] = current_seq.numpy()
        inputs["position_ids"] = (
            torch.arange(inputs["input_ids"].shape[1], dtype=torch.long)
            .unsqueeze(0)
            .numpy()
        )
        inputs["attention_mask"] = torch.ones(
            [
                1,
                inputs[f"past_key_values.0.key"].shape[2]
                + inputs["input_ids"].shape[1],
            ],
            dtype=torch.int64,
        ).numpy()

        # Run the model with MAX engine
        max_outputs = maxmodel.execute(**inputs)
        outputs = torch.from_numpy(max_outputs["logits"])

        # Get the newly generated next token
        next_token_logits = outputs[:, -1, :]
        next_tokens_scores = logits_processor(current_seq, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # Append the new token to our sequence
        current_seq = torch.cat([current_seq, next_tokens[:, None]], dim=-1)

        # Update the KV cache for the next iteration
        for i in range(N_LAYERS):
            inputs[f"past_key_values.{i}.key"] = max_outputs[f"present.{i}.key"]
            inputs[f"past_key_values.{i}.value"] = max_outputs[
                f"present.{i}.value"
            ]

    return current_seq.numpy()


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--text",
        type=str,
        metavar="str",
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

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    torch.set_default_device("cpu")

    # Preprocess input statement
    print("Processing input...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    except OSError:
        print(HUGGINGFACE_ERROR)
        exit(1)

    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(
        args.text, return_tensors="pt", max_length=128, truncation=True
    ).input_ids
    print("Input processed.\n")

    outputs = generate(args.model_path, input_ids)
    print("Text generated.\n")

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][
        len(args.text) :
    ]

    print(f"Prompt: {args.text}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
