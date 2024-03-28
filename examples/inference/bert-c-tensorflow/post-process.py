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

import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

HF_MODEL = "bert-base-uncased"


def post_process():
    parser = ArgumentParser(description="Input given to the model")
    parser.add_argument(
        "--input",
        type=str,
        metavar="<text>",
        required=True,
        help="Masked input.",
    )
    args = parser.parse_args()

    # Extract label prediction from output
    print("Extracting class from outputs...")
    logits = np.fromfile("outputs.bin", dtype=np.float32).reshape((1, 9, 30522))

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    inputs = tokenizer(args.input, return_tensors="tf")
    mask_token_index = tf.where(
        (inputs.input_ids == tokenizer.mask_token_id)[0]
    )
    selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)
    predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
    prediction = tokenizer.decode(predicted_token_id)

    print(f"The prediction is: {prediction}")


def cleanup():
    inputs_dir = Path.cwd() / "inputs"
    shutil.rmtree(inputs_dir, ignore_errors=True)

    build_dir = Path.cwd() / "build"
    shutil.rmtree(build_dir, ignore_errors=True)

    outputs = Path.cwd() / "outputs.bin"
    outputs.unlink(missing_ok=True)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    post_process()
    cleanup()
