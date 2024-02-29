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
import pickle
import common
import numpy as np
import sys

model_name = sys.argv[1]
loaded = None
inputs = None

session = engine.InferenceSession()
model = session.load(f"./.cache/{model_name}_savedmodel") # Load TensorFlow model

if model_name == "roberta":
    with open(".cache/roberta.pkl", "rb") as f:
        inputs = pickle.load(f)
elif model_name == "clip":
    with open(".cache/clip.pkl", "rb") as f:
        inputs = pickle.load(f)
        inputs = {
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int32),
            "input_ids": inputs["input_ids"].numpy().astype(np.int32),
            "pixel_values": inputs["pixel_values"].numpy().astype(np.float32)
        }

qps = common.run(lambda: model.execute(**inputs))
common.save_result("max", qps)
