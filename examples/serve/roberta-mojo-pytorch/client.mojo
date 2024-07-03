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

from max.engine import InferenceSession
from max.serve.kserve.client import GRPCClient
from python_utils import numpy_to_tensor, tensor_to_numpy
from python import Python


def main():
    transformers = Python.import_module("transformers")
    np = Python.import_module("numpy")

    HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
    hf_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME
    )
    hf_model.config.return_dict = False

    # Tokenize input into input ids and mask:
    INPUT = (
        "There are many exciting developments in the field of AI"
        " Infrastructure!"
    )
    SEQ_LEN = 128
    tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    raw_inputs = tokenizer(
        INPUT,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
    )

    # Prepare the inputs
    raw_input_ids = raw_inputs["input_ids"].detach().numpy()
    raw_attention_mask = raw_inputs["attention_mask"].detach().numpy()
    input_ids = numpy_to_tensor[DType.int64](raw_input_ids)
    attention_mask = numpy_to_tensor[DType.int64](raw_attention_mask)

    session = InferenceSession()
    inputs = session.new_tensor_map()
    inputs.borrow("input_ids", input_ids)
    inputs.borrow("attention_mask", attention_mask)
    for key in inputs.keys():
        print(key[] + " : " + str(inputs.get[DType.int64](key[])))

    # Run inference
    req_outputs = List[String]("result0")
    client = GRPCClient("0.0.0.0:8000", session)
    response = client.infer("roberta", "0", inputs, req_outputs)
    outputs = response.get_output_tensors()
    for key in outputs.keys():
        print(key[] + " : " + str(outputs.get[DType.float32](key[])))

    _ = raw_input_ids^
    _ = raw_attention_mask^
    _ = input_ids^
    _ = attention_mask^

    arr = tensor_to_numpy(outputs.get[DType.float32]("result0"), np)

    # Extract class prediction from output
    predicted_class_id = arr.argmax(axis=-1)[0]
    classification = hf_model.config.id2label[predicted_class_id]

    print("The sentiment is: " + str(classification))
