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

#!/usr/bin/env python3

import common
import os
import tensorflow as tf
from pathlib import Path
import transformers
from transformers import TFRobertaForSequenceClassification, TFCLIPModel
import pickle
import sys

transformers.utils.logging.set_verbosity_error()


model_name = sys.argv[1]
model_dir =  f"./.cache/{model_name}_savedmodel"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = Path(script_dir, model_dir)

if model_name == "roberta":
    if not os.path.exists(model_path):
        model = TFRobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest")
        tf.saved_model.save(model, model_dir)

    with open(".cache/roberta.pkl", "rb") as f:
        inputs = pickle.load(f)
        inputs = {
            "attention_mask": tf.convert_to_tensor(
                inputs["attention_mask"], dtype=tf.int32
            ),
            "input_ids": tf.convert_to_tensor(inputs["input_ids"], dtype=tf.int32),
            "token_type_ids": tf.convert_to_tensor(
                inputs["token_type_ids"], dtype=tf.int32
            ),
        }
elif model_name == "clip":
    if not os.path.exists(model_path):
        model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        tf.saved_model.save(model, model_dir)

    with open(".cache/clip.pkl", "rb") as f:
        inputs = pickle.load(f)
        inputs = {
            "attention_mask": tf.convert_to_tensor(
                inputs["attention_mask"].numpy(), dtype=tf.int32
            ),
            "input_ids": tf.convert_to_tensor(inputs["input_ids"].numpy(), dtype=tf.int32),
            "pixel_values": tf.convert_to_tensor(
                inputs["pixel_values"].numpy(), dtype=tf.float32
            ),
        }

loaded = tf.saved_model.load(model_dir)
qps = common.run(lambda: loaded(inputs))
common.save_result("tf", qps)

# Prevents an ugly error on unload
del loaded
