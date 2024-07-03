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

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
hf_model.config.return_dict = False

# Converting model to TorchScript
model_path = Path("roberta.torchscript")
batch = 1
seqlen = 128
inputs = {
    "input_ids": torch.zeros((batch, seqlen), dtype=torch.int64),
    "attention_mask": torch.zeros((batch, seqlen), dtype=torch.int64),
}
with torch.no_grad():
    traced_model = torch.jit.trace(
        hf_model, example_kwarg_inputs=inputs, strict=False
    )

torch.jit.save(traced_model, model_path)
