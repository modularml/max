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
import pickle

import common
import numpy as np
from max import engine


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
    inputs = None

    session = engine.InferenceSession()

    if args.model == "roberta":
        with open(".cache/roberta.pkl", "rb") as f:
            inputs = pickle.load(f)
        input_spec_list = [
            engine.TorchInputSpec(shape=(1, None), dtype=engine.DType.int64),
            engine.TorchInputSpec(shape=(1, None), dtype=engine.DType.int64),
            engine.TorchInputSpec(shape=(1, None), dtype=engine.DType.int64),
        ]
    elif args.model == "clip":
        with open(".cache/clip.pkl", "rb") as f:
            inputs = pickle.load(f)
            inputs = {
                "attention_mask": inputs["attention_mask"]
                .numpy()
                .astype(np.int32),
                "input_ids": inputs["input_ids"].numpy().astype(np.int32),
                "pixel_values": inputs["pixel_values"]
                .numpy()
                .astype(np.float32),
            }
        TEXT_BATCH = 2
        SEQLEN = 7
        IMAGE_BATCH = 1
        CHANNELS = 3
        IMAGE_SIZE = 224
        input_spec_list = [
            engine.TorchInputSpec(
                shape=(TEXT_BATCH, SEQLEN), dtype=engine.DType.int64
            ),
            engine.TorchInputSpec(
                shape=(IMAGE_BATCH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                dtype=engine.DType.float32,
            ),
            engine.TorchInputSpec(
                shape=(TEXT_BATCH, SEQLEN), dtype=engine.DType.int64
            ),
        ]

    model = session.load(
        f"./.cache/{args.model}.torchscript", input_specs=input_spec_list
    )  # Load PyTorch model

    qps = common.run(lambda: model.execute(**inputs))
    common.save_result("max", qps)


if __name__ == "__main__":
    main()
