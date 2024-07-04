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
import streamlit as st
import select
from shared import menu

st.set_page_config("Bert", page_icon="👓")

"""
# 👓 Bert

A basic implementation of Bert using MAX. Type a text string, using `[MASK]` to indicate where you want the model to predict a word.
"""

menu()

model_state = st.empty()

# MODEL_PATH = "bert-mlm.torchscript"
# MODEL_DIR = "bert-mlm"
# INPUT_EXAMPLE = "Paris is the [MASK] of France."
# HF_MODEL_NAME = "bert-base-uncased"
# OUTPUT_PATH = "./models/bert-mlm.torchscript"

def download_model(venv_location, debug=False):

    # set venv
    #venv = "/home/ubuntu/max-nightly-venv/bin/python"

    venv = venv_location + "/bin/python"
    venv = os.path.expandvars(venv)

    # set path for simple_inference.py
    script_path = "../inference/common/bert-torchscript/download-model.py"

    # call the script.
    process = subprocess.Popen([venv,
                                script_path,
                                "-o",
                                model_path,
                                "--mlm"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
    
    if debug:
        # Read and display the output and errors in real-time
        output = ""
        errors = ""

        # Use select to handle process output in real-time
        while True:
            # Wait for any of the streams to have data
            reads, _, _ = select.select([process.stdout, process.stderr], [], [])

            for stream in reads:
                # Read a line from stdout
                if stream == process.stdout:
                    output_line = process.stdout.readline()
                    if output_line:
                        output += output_line
                        st.text(output_line.strip())

                # Read a line from stderr
                elif stream == process.stderr:
                    error_line = process.stderr.readline()
                    if error_line:
                        errors += error_line
                        st.text(error_line.strip())

            # Check if the process has finished
            if process.poll() is not None:
                break

        # Ensure all remaining lines are read
        remaining_output = process.stdout.read()
        if remaining_output:
            output += remaining_output
            st.text(remaining_output.strip())

        remaining_errors = process.stderr.read()
        if remaining_errors:
            errors += remaining_errors
            st.text(remaining_errors.strip())

def simple_inference(text, venv_location, debug=False):

    if not validate_text(text):
        return "You must add [MASK] to your text string."
    
    # Download the model. 
    download_model(venv_location, False)

    # set venv
    #venv = "/home/ubuntu/max-nightly-venv/bin/python"
    venv = venv_location + "/bin/python"
    venv = os.path.expandvars(venv)

    # set path for simple_inference.py
    script_path = "../inference/bert-python-torchscript/simple-inference.py"

    # call the script.
    process = subprocess.Popen([venv,
                                script_path,
                                "--model-path",
                                model_path,
                                "--text",
                                text],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
    
    filled_text_line = None
    output = ""
    errors = ""

    # Use select to handle process output in real-time
    while True:
        # Wait for any of the streams to have data
        reads, _, _ = select.select([process.stdout, process.stderr], [], [])

        for stream in reads:
            # Read a line from stdout
            if stream == process.stdout:
                output_line = process.stdout.readline()
                if output_line:
                    output += output_line
                    if debug:
                        st.text(output_line.strip())
                    
                    # Check if the line starts with "filled text"
                    if output_line.startswith("filled mask: "):
                        filled_text_line = output_line.strip()

            # Read a line from stderr
            elif stream == process.stderr:
                error_line = process.stderr.readline()
                if error_line:
                    errors += error_line
                    if debug:
                        st.text(error_line.strip())

        # Check if the process has finished
        if process.poll() is not None:
            break

    # Ensure all remaining lines are read
    remaining_output = process.stdout.read()
    if remaining_output:
        output += remaining_output
        if debug:
            st.text(remaining_output.strip())
        
        # Check remaining output for "filled text"
        if filled_text_line is None:
            for line in remaining_output.splitlines():
                if line.startswith("filled mask:"):
                    filled_text_line = line.strip()
                    break

    remaining_errors = process.stderr.read()
    if remaining_errors:
        errors += remaining_errors
        if debug:
            st.text(remaining_errors.strip())
    return filled_text_line

def validate_text(text):
    return "[MASK]" in text


# Streamlit UI

# sidebar
venv_location = st.sidebar.text_input("Python environment (venv) location", value="$HOME/max-nightly-venv")
model_name = st.sidebar.text_input("Model Filename", value="bert-mlm.torchscript")
model_path = os.path.join("models", os.path.basename(model_name))
model_path = os.path.abspath(model_path)

# Initialize session state input text boxes if text doesn't exist
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

if 'output_text' not in st.session_state:
    st.session_state.output_text = ""

# Input text box
st.session_state.input_text = st.text_input("Input text:", value=st.session_state.input_text)

if st.button("Run inference"):
    filled_text = simple_inference(st.session_state.input_text, venv_location)
    st.session_state.output_text = filled_text.replace("filled mask: ", "", 1)

st.text_area("Output text:", value=st.session_state.output_text, height=50)
