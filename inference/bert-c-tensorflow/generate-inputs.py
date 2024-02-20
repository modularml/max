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
from transformers import AutoTokenizer

HF_MODEL_NAME = "bert-base-uncased"


def main():
    # Parse args
    parser = ArgumentParser(description="Generate inputs for model")
    parser.add_argument(
        "--input",
        type=str,
        metavar="<text>",
        required=True,
        help="Statement to classify.",
    )
    args = parser.parse_args()

    # Preprocess input statement
    print("Processing input...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    inputs = tokenizer(args.input, return_tensors="np")
    inputs["token_type_ids"] = inputs.get(
        "token_type_ids", np.zeros_like(inputs["input_ids"])
    )
    print("Input processed.\n")

    print("Saving inputs to disk...")
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True)
    print(f"Saving inputs to {input_dir}...")
    for name, value in inputs.items():
        value = value.astype(np.int32)
        filename = input_dir / name
        filename = filename.with_suffix(".bin")
        filename.unlink(missing_ok=True)
        value.tofile(filename)

        shape = np.array(value.shape).astype(np.int64)
        shape_file = input_dir / f"{name}_shape.bin"
        shape.tofile(shape_file)


if __name__ == "__main__":
    main()
