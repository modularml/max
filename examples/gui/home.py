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

import streamlit as st

from shared import menu

st.set_page_config("MAX", "⚡️")

menu()
"""# MAX ⚡️ Examples

Welcome to MAX! Select an example to get started:
"""

if st.button("🦙 Llama3"):
    st.switch_page("pages/llama3.py")
elif st.button("👓 BERT"):
    st.switch_page("pages/bert.py")

