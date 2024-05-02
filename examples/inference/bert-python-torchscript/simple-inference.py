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

from argparse import ArgumentParser

import signal
import torch
from transformers import BertTokenizer

# suppress extraneous logging
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BATCH = 1
SEQLEN = 128
DEFAULT_MODEL_PATH = "../../models/bert-mlm.torchscript"
DESCRIPTION = "BERT model"
HF_MODEL_NAME = "bert-base-uncased"


def execute(model_path, text, input_dict):
    session = engine.InferenceSession()
    input_spec_list = [
        engine.TorchInputSpec(shape=tensor.size(), dtype=engine.DType.int64)
        for tensor in input_dict.values()
    ]
    model = session.load(model_path, input_specs=input_spec_list)
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
    print("Processing input...")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQLEN,
    )
    print("Input processed.\n")
    masked_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]
    outputs = model.execute(**inputs)["result0"]
    logits = torch.from_numpy(outputs[0, masked_index, :])
    predicted_token_id = logits.argmax(dim=-1)
    predicted_tokens = tokenizer.decode(
        [predicted_token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return predicted_tokens


def main():
    # Parse args
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--text",
        type=str,
        metavar="<text>",
        required=True,
        help="Masked language model.",
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
    input_dict = {
        "input_ids": torch.zeros((BATCH, SEQLEN), dtype=torch.int64),
        "attention_mask": torch.zeros((BATCH, SEQLEN), dtype=torch.int64),
        "token_type_ids": torch.zeros((BATCH, SEQLEN), dtype=torch.int64),
    }

    outputs = execute(args.model_path, args.text, input_dict)
    # Get the predictions for the masked token
    print(f"input text: {args.text}")
    print(f"filled mask: {args.text.replace('[MASK]', outputs)}")


if __name__ == "__main__":
    main()
