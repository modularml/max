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

import common
import argparse
import pickle
import sys
import numpy as np
from transformers import AutoProcessor
from PIL import Image
import requests


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
    if args.model == "roberta":
        rng = np.random.default_rng()
        inputs = {
            "input_ids": rng.integers(
                low=0, high=50264, size=(1, 128), dtype=np.int64
            ),
            "token_type_ids": np.zeros((1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        }
        with open(".cache/roberta.pkl", "wb") as f:
            pickle.dump(dict(**inputs), f)
    elif args.model == "clip":
        processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with open(".cache/clip.pkl", "wb") as f:
            pickle.dump(dict(**inputs), f)


if __name__ == "__main__":
    main()
