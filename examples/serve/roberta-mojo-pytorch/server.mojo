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

from max.tensor import TensorSpec
from max.engine import InferenceSession, InputSpec
from max.serve.kserve import FileModel, GRPCServer, MuxInferenceService


def main():
    model_name = "roberta"
    model_path = "roberta.torchscript"

    # Load models during service creation:
    models = List(
        FileModel(
            model_name,
            "0",
            model_path,
            List(
                InputSpec(TensorSpec(DType.int64, 1, 128)),
                InputSpec(TensorSpec(DType.int64, 1, 128)),
            ),
        )
    )
    session = InferenceSession()
    service = MuxInferenceService(models, session)

    # Create service and start listening:
    server = GRPCServer.create("0.0.0.0:8000", session)
    print("Listening on port 8000!")
    server.serve(service)
