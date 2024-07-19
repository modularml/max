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
import os
import subprocess

import openai
import streamlit as st
import chromadb
from fastembed import TextEmbedding
from llama_index.core import SimpleDirectoryReader

from shared import download_file, kill_process, menu, modular_cache_dir

st.set_page_config("RAG with Llama3", page_icon="🗣️ 📄")

menu()

os.environ["TOKENIZERS_PARALLELISM"] = "0"

"""
# 🗣️ 📄 Retrieval Augmented Generation with MAX Pipeline Llama3

Place your local data under `examples/gui/ragdata` sub-repository to be indexed.
Supported formats: `.txt, .pdf, .csv, .docx, .epub, .ipynb, .md, .html`.

For LLM, select a quantization encoding to download model from a predefined `Model URL`.
If the model exists at `Model Path` it won't be downloaded again.
You can set a custom `Model URL` or `Model Path` that matches the quantization encoding.
"""

SYSTEM_PROMPT = st.sidebar.text_area(
    "System Prompt",
    value="""You are a helpful document search assistant.
    Your task is to find an answer to user's QUERY about their given documentations.
    Be helpful and concise.
    Think step by step.
    DO NOT HALLUCINATE.""",
)

QA_PROMPT = """Here is the context:
Your are given a list pairs of text documents and their sources as CONTEXT {data}.
Find the most relevant document that matches the QUERY {query} and give a detailed `ANSWER` by including the `SOURCE` filename.
You are allowed to show an relevant code from the context. In case you don't the answer say 'I don't know!'
## EXAMPLE:
CONTEXT: [('MAX graph api enables building computational graphs with ease', 'graph-api.md'), ('Install MAX with `modular install max`', 'install.md')]
QUERY: "MAX graph api"
ANSWER: 'MAX graph api enables building computational graphs with ease', SOURCE: graph-api.md
## GO"""

DATA_PATH = Path(os.path.dirname(os.path.dirname(__file__))) / "data"
model_state = st.empty()


@st.cache_resource(show_spinner=False)
def load_embed_docs():
    docs = SimpleDirectoryReader(
        "./ragdata", exclude=[".gitkeep"], recursive=True
    ).load_data()
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "max-rag-example", metadata={"hnsw:space": "cosine"}
    )
    embedding_model = TextEmbedding()

    for i, doc in enumerate(docs):
        embedding = list(embedding_model.embed(doc.text))[0].tolist()
        collection.upsert(
            documents=doc.text,
            embeddings=embedding,
            ids=[str(i)],
            metadatas=[doc.metadata],
        )

    return collection, embedding_model


collection, embedding_model = load_embed_docs()
model_state.success("Data is indexed", icon="✅")


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
            "--prompt",
            "start",
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
    model_url = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q4_0.gguf"
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

n_result = st.sidebar.slider("Number of Top Embedding Search Results", 1, 7, 5)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Questions about the your Docs"):
    st.session_state.messages.append(
        {"role": "user", "avatar": "💬", "content": prompt}
    )

    with st.chat_message("user", avatar="💬"):
        st.markdown(prompt)

    query_embedding = list(embedding_model.embed(prompt))[0].tolist()
    ret = collection.query(query_embedding, n_results=n_result)
    data = []
    for i, (doc, metadata) in enumerate(
        zip(ret["documents"], ret["metadatas"])
    ):
        data.append(("\n\n".join(doc), metadata[0]["file_name"]))

    with st.chat_message("assistant", avatar="🦙") and st.spinner(
        "Thinking ..."
    ):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [
            {
                "role": "user",
                "content": QA_PROMPT.format(**{"data": data, "query": prompt}),
            }
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
