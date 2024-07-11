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
import subprocess

import openai
import streamlit as st
from shared import download_file, kill_process, menu, modular_cache_dir

st.set_page_config("Llama3", page_icon="🦙")
menu()

"""
# 🦙 Llama3

Select a quantization encoding to download model from a predefined `Model URL`. If the model exists at `Model Path` it won't be downloaded again. You can set a custom `Model URL` or `Model Path` that matches the quantization encoding.
"""

model_state = st.empty()


@st.cache_resource(show_spinner=False)
def start_llama3(
    temperature,
    max_tokens,
    min_p,
    custom_ops_path,
    tokenizer_path,
    quantization,
    model_path,
):
    with st.spinner("Starting Llama3"):
        model_state = st.empty()
        # Kill server if it's already running
        kill_process(8000, model_state)
        command = [
            "mojo",
            "run",
            "../graph-api/serve_pipeline.🔥",
            "llama3",
            "--max-tokens",
            str(max_tokens),
            "--model-path",
            model_path,
            "--quantization-encoding",
            quantization,
            "--temperature",
            str(temperature),
            "--min-p",
            str(min_p),
        ]
        if custom_ops_path:
            command.append("--custom-ops-path")
            command.append(custom_ops_path)
        if tokenizer_path:
            command.append("--tokenizer-path")
            command.append(tokenizer_path)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if process.stderr:
            model_state.error(process.stderr, icon="🚨")

        # Generator to yield characters from the subprocess output
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                if line.strip() == "Listening on port 8000!":
                    model_state.success("Llama3 is ready!", icon="✅")
                    return
                if line.strip().startswith("mojo: error:"):
                    model_state.error(line, icon="🚨")
                    exit(1)
                else:
                    model_state.info(line, icon="🛠️")


quantization = st.sidebar.selectbox(
    "Quantization Encoding", ["q4_k", "q4_0", "q6_k"]
)

if quantization == "q4_0":
    model_url = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
elif quantization == "q4_k":
    model_url = "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
elif quantization == "q6_k":
    model_url = "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf"

model_url = st.sidebar.text_input("Model URL", value=model_url)
os.makedirs(modular_cache_dir(), exist_ok=True)
model_path = os.path.join(modular_cache_dir(), os.path.basename(model_url))
model_path = st.sidebar.text_input("Model Path", value=model_path)

download_file(model_url, model_path, model_state)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 0, 8192, 8192)
min_p = st.sidebar.slider("Minimum Probability Threshold", 0.0, 1.0, 0.05)
custom_ops_path = st.sidebar.text_input("Custom Ops Path")
tokenizer_path = st.sidebar.text_input("Tokenizer Path")
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a helpful coding assistant named MAX Llama3.",
)
start_local_server = st.sidebar.checkbox("Start Local Server", True)
if start_local_server:
    server_ip_address = st.sidebar.text_input(
        "Server Address", "http://localhost:8000", disabled=True
    )
else:
    server_ip_address = st.sidebar.text_input(
        "Server Address", "http://localhost:8000", disabled=False
    )
client = openai.OpenAI(api_key="NA", base_url=f"{server_ip_address}/v1")

if start_local_server:
    start_llama3(
        temperature,
        max_tokens,
        min_p,
        custom_ops_path,
        tokenizer_path,
        quantization,
        model_path,
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message to llama3"):
    st.session_state.messages.append(
        {"role": "user", "avatar": "💬", "content": prompt}
    )

    with st.chat_message("user", avatar="💬"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦙"):
        messages = [{"role": "system", "content": system_prompt}]
        messages += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        stream = client.chat.completions.create(
            model="",
            messages=messages,  # type: ignore
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "avatar": "🦙"}
    )
