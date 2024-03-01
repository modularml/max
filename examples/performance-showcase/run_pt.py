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

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch not found. Please python3 -m pip install torch to get results!")
    exit(1)

import torch
import os
from pathlib import Path
import pickle
import sys
from transformers import RobertaForSequenceClassification, CLIPModel

model_name = sys.argv[1]
model_dir =  f"./.cache/{model_name}_pt"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = Path(script_dir, model_dir)


if model_name == "roberta":
    if not os.path.exists(model_path):
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest")
        torch.save(model, model_dir)

    loaded = torch.load(model_dir)
    with open(".cache/roberta.pkl", "rb") as f:
        inputs = pickle.load(f)
        inputs = [
            torch.from_numpy(inputs["input_ids"]),
            torch.from_numpy(inputs["attention_mask"]),
            torch.from_numpy(inputs["token_type_ids"]),
        ]

    with torch.inference_mode():
        qps = common.run(lambda: loaded.forward(*inputs))
    common.save_result("pt", qps)
elif model_name == "clip":
    if not os.path.exists(model_path):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        torch.save(model, model_dir)

    loaded = torch.load(model_dir)
    with open(".cache/clip.pkl", "rb") as f:
        inputs = pickle.load(f)

    with torch.inference_mode():
        qps = common.run(lambda: loaded.forward(**inputs))
    common.save_result("pt", qps)
