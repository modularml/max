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

from .interfaces import TokenGenerator
from utils import TextGenerationMetrics


async def stream_text_to_console(
    model: TokenGenerator, prompt: str, metrics: Optional[TextGenerationMetrics]
):
    context = await model.new_context(prompt)
    prompt_size = context.prompt_size
    max_tokens = context.max_tokens

    # Start with the initial prompt.
    print(context.prompt, end="", flush=True)
    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    # Note: assume a single request for now.
    request_id = 0
    for i in range(prompt_size, max_tokens + 1):
        response = await model.next_token({request_id: context})
        if request_id not in response:
            break
        if metrics:
            if i == prompt_size:
                metrics.signpost("first_token")
            metrics.new_token()
        print(response[request_id], end="", flush=True)
    if metrics:
        metrics.signpost("end_generation")
    print()
