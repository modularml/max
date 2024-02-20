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
import pickle
import sys
from transformers import AutoTokenizer, TFRobertaForSequenceClassification, AutoProcessor
from PIL import Image
import requests


model_name = sys.argv[1]
if model_name == "roberta":
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest")
    inputs = tokenizer(
        "I am a little teapot", return_tensors="np", return_token_type_ids=True
    )
    with open(".cache/roberta.pkl", "wb") as f:
        pickle.dump(dict(**inputs), f)
elif model_name == "clip":  
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
    )
    with open(".cache/clip.pkl", "wb") as f:
        pickle.dump(dict(**inputs), f)

