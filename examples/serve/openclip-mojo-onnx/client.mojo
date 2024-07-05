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
    # Import Python libraries
    open_clip = Python.import_module("open_clip")
    PIL = Python.import_module("PIL")
    requests = Python.import_module("requests")
    np = Python.import_module("numpy")

    tup = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Prepare input
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    labels = ["cats", "dogs", "fish"]
    raw_image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = tup[2](raw_image).unsqueeze(0).detach().numpy()
    text = tokenizer(labels).detach().numpy()

    session = InferenceSession()
    inputs = session.new_tensor_map()
    image_tensor = numpy_to_tensor[DType.float32](image)
    text_tensor = numpy_to_tensor[DType.int64](text)
    inputs.borrow("image", image_tensor)
    inputs.borrow("text", text_tensor)
    print(str(inputs))

    # Run inference
    req_outputs = List[String]("image_features", "text_features")
    client = GRPCClient("0.0.0.0:8000", session)
    response = client.infer("openclip", "0", inputs, req_outputs)
    outputs = response.get_output_tensors()

    _ = image^
    _ = text^
    _ = image_tensor^
    _ = text_tensor^

    img_feats = tensor_to_numpy(
        outputs.get[DType.float32]("image_features"), np
    )
    txt_feats = tensor_to_numpy(outputs.get[DType.int64]("text_features"), np)

    def softmax(np: PythonObject, x: PythonObject) -> PythonObject:
        z = x - np.max(x)
        num = np.exp(z)
        return np.exp(z) / np.sum(num)

    txt_feats /= np.linalg.norm(txt_feats)
    img_feats /= np.linalg.norm(img_feats)
    similarity = softmax(np, 100.0 * np.matmul(img_feats, txt_feats.T))
    print("Label probs:\n", similarity)
