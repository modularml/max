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

from typing import Optional

from utils import TextGenerationMetrics

from .interfaces import TokenGenerator


async def stream_text_to_console(
    model: TokenGenerator,
    prompt: str,
    metrics: Optional[TextGenerationMetrics] = None,
):
    context = await model.new_context(prompt)
    prompt_size = len(context.tokens)

    # Start with the initial prompt.
    print(prompt, end="", flush=True)
    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    # Note: assume a single request for now.
    is_first_token = True
    request_id = str(id(prompt))
    while True:
        response = await model.next_token({request_id: context})
        if request_id not in response:
            break
        if metrics:
            if is_first_token:
                is_first_token = False
                metrics.signpost("first_token")
            metrics.new_token()
        print(response[request_id], end="", flush=True)
    if metrics:
        metrics.signpost("end_generation")
    print()
