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
from pathlib import Path

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer, logging


HF_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_PATH = "bert.torchscript"


def main():
    parser = ArgumentParser(description="Download model for inference.")
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Location to save the model",
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--text",
        type=str,
        metavar="<text>",
        required=True,
        help="Statement to classify.",
    )

    args = parser.parse_args()

    model_path = Path(args.output_path)

    print("Downloading model...")
    logging.set_verbosity_error()  # Disable warning suggesting to train the model
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    model.eval()

    print("Generating input tensors...")
    print(f'Input sentence: "{args.text}".')
    encoded_inputs = tokenizer(args.text, return_tensors="pt")

    print("Saving inputs to disk...")
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True)

    created_files = []
    for name, value in encoded_inputs.items():
        value = value.numpy().astype(np.int32)
        filename = input_dir / name
        filename = filename.with_suffix(".bin")
        filename.unlink(missing_ok=True)
        value.tofile(filename)

        shape = np.array(value.shape).astype(np.int64)
        shape_file = input_dir / f"{name}_shape.bin"
        shape.tofile(shape_file)
        created_files += [str(filename), str(shape_file)]
    print("Inputs saved.")

    print("Saving model in TorchScript format...")
    model_path = Path(args.output_path)
    if model_path.exists():
        print(f"'{args.output_path}' already exists.\n")
    else:
        print("Converting the model to TorchScript format...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, example_kwarg_inputs=dict(encoded_inputs), strict=False
            )

        torch.jit.save(traced_model, model_path)
        created_files += [str(model_path)]
        print(f"Model saved.")

    print(
        "\nCreated/updated following files:\n   %s\n"
        % "\n   ".join(created_files)
    )


if __name__ == "__main__":
    main()