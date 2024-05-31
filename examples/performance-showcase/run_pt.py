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

import argparse

import common

try:
    import torch
except ModuleNotFoundError:
    print(
        "PyTorch not found. Please python3 -m pip install torch to get results!"
    )
    exit(1)

import os
import pickle

import torch
from transformers import CLIPModel, RobertaForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=["roberta", "clip"],
        help="Choose from one of these models",
        required=True,
    )
    args = parser.parse_args()
    model_script = f"./.cache/{args.model}.torchscript"

    if args.model == "roberta":
        with open(".cache/roberta.pkl", "rb") as f:
            inputs = pickle.load(f)
            inputs = {
                "input_ids": torch.from_numpy(inputs["input_ids"]),
                "attention_mask": torch.from_numpy(inputs["attention_mask"]),
                "token_type_ids": torch.from_numpy(inputs["token_type_ids"]),
            }

        model = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
        )
        if not os.path.exists(model_script):
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model, example_kwarg_inputs=dict(inputs), strict=False
                )
            torch.jit.save(traced_model, model_script)

        with torch.inference_mode():
            qps = common.run(lambda: model.forward(**inputs))
        common.save_result("pt", qps)
    elif args.model == "clip":
        with open(".cache/clip.pkl", "rb") as f:
            inputs = pickle.load(f)

        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torchscript=True
        )
        if not os.path.exists(model_script):
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model, example_kwarg_inputs=dict(inputs), strict=False
                )

            torch.jit.save(traced_model, model_script)

        with torch.inference_mode():
            qps = common.run(lambda: model.forward(**inputs))
        common.save_result("pt", qps)


if __name__ == "__main__":
    main()
