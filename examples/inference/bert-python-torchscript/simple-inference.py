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

from max import engine

DEFAULT_MODEL_PATH = "bert.torchscript"
DESCRIPTION = "BERT model"


def execute(model_path, inputs, input_shapes):
    session = engine.InferenceSession()

    input_spec_list = []
    for _, inp in input_shapes.items():
        input_spec_list.append(
            engine.TorchInputSpec(shape=inp, dtype=engine.DType.int64)
        )

    options = engine.TorchLoadOptions(input_spec_list)

    print("Loading model...")
    modular_model = session.load(model_path, options)
    print("Model loaded.\n")

    print("Executing model...")
    outputs = modular_model.execute(
        attention_mask=inputs["attention_mask"],
        input_ids=inputs["input_ids"],
        token_type_ids=inputs["token_type_ids"],
    )
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

    input_dict = dict()
    input_shape_dict = dict()

    input_dir = Path("inputs")
    for input_key in ["input_ids", "attention_mask", "token_type_ids"]:
        input_file = input_dir / f"{input_key}.bin"
        curr_input = np.fromfile(input_file, dtype=np.int32)

        # Add a batch dim before storing them as inputs.
        batched_curr_input = curr_input[np.newaxis, :]
        input_dict[input_key] = batched_curr_input

        input_shape_file = input_dir / f"{input_key}_shape.bin"
        curr_input_shape = np.fromfile(input_shape_file, dtype=np.int64)
        input_shape_dict[input_key] = curr_input_shape

    # Classify input statement
    outputs = execute(
        model_path=args.model_path,
        inputs=input_dict,
        input_shapes=input_shape_dict,
    )

    outputs = outputs["result0"]  # Unwrap the outermost "result0" output dict.
    logits = np.array(outputs["logits"]).astype(np.float32)
    logits.tofile("outputs.bin")

if __name__ == "__main__":
    main()