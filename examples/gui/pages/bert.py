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

import os

import streamlit as st
import torch
from max import engine
from shared import menu, modular_cache_dir
from transformers import (
    AutoModelForSequenceClassification,
    BertForMaskedLM,
    BertTokenizer,
)

st.set_page_config("Bert", page_icon="👓")
menu()

"""
# 👓 Bert

A basic implementation of Bert using MAX. Type a text string, using `[MASK]` to indicate where you want the model to predict a word.
"""

HF_MODEL_NAME = "bert-base-uncased"


# If batch, seq_len, or mlm options change, recompile the torchscript.
@st.cache_data
def compile_torchscript(batch: int, seq_len: int, mlm: bool):
    if mlm:
        model = BertForMaskedLM.from_pretrained(HF_MODEL_NAME)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_NAME
        )
    input_dict = {
        "input_ids": torch.zeros((batch, seq_len), dtype=torch.int64),
        "attention_mask": torch.zeros((batch, seq_len), dtype=torch.int64),
        "token_type_ids": torch.zeros((batch, seq_len), dtype=torch.int64),
    }
    model.eval()
    model.config.return_dict = False
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, example_kwarg_inputs=dict(input_dict), strict=False
        )
    torch.jit.save(traced_model, model_path)


model_state = st.empty()

mlm = st.sidebar.checkbox("Masked Language Model", True)
filename = "bert.torchscript"
if mlm:
    filename = "bert-mlm.torchscript"
model_path = st.sidebar.text_input(
    "Model Path",
    os.path.join(modular_cache_dir(), filename),
)
batch = st.sidebar.number_input("Batch Size", 1, 64)
seq_len = st.sidebar.slider("Sequence Length", 128, 1024)
input_text = st.text_input("Text Input", "Don't [MASK] about it")


compile_torchscript(batch, seq_len, mlm)


if st.button("Predict Word"):
    masks = input_text.split("[MASK]")
    if len(masks) > 2:
        st.error("Cannot have more than a single [MASK] in the input text")
        exit(1)
    if len(masks) < 2:
        st.error("Require at least one [MASK] in the input text")
        exit(1)

    session = engine.InferenceSession()
    inputs = [
        torch.zeros((batch, seq_len), dtype=torch.int64),
        torch.zeros((batch, seq_len), dtype=torch.int64),
        torch.zeros((batch, seq_len), dtype=torch.int64),
    ]
    input_spec_list = [
        engine.TorchInputSpec(shape=tensor.size(), dtype=engine.DType.int64)
        for tensor in inputs
    ]
    model = session.load(model_path, input_specs=input_spec_list)
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    print("Input processed.\n")
    masked_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]
    outputs = model.execute(**inputs)["result0"]
    logits = torch.from_numpy(outputs[0, masked_index, :])
    predicted_token_id = logits.argmax(dim=-1)
    predicted_tokens = tokenizer.decode(
        [predicted_token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    st.text_area(
        "Filled Mask", input_text.replace("[MASK]", predicted_tokens, 40)
    )
