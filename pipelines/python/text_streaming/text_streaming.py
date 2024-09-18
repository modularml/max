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

import uuid
from typing import Optional

from utils import TextGenerationMetrics

from .interfaces import TokenGenerator


async def stream_text_to_console(
    model: TokenGenerator,
    prompt: str,
    metrics: Optional[TextGenerationMetrics] = None,
    max_batch_size: int = 1,
):
    # Length of is_first_token and request_id_context_dict should be == batch_size.
    is_first_token: dict[str, bool] = {}
    request_id_context_dict = dict()

    # TODO(MSDK-972): Make this batch_size variable based on size of the request dict.
    # NOTE: This batch_size param also needs to be == config.batch_size of
    # the underlying pipeline config.
    batch_size = max_batch_size

    # create a dict of request_id: contexts
    for _ in range(batch_size):
        # We make the key unique even for the same prompts for now.
        req_id = str(uuid.uuid4())
        context = await model.new_context(prompt)
        request_id_context_dict[req_id] = context
        is_first_token[req_id] = True
        prompt_size = len(context.tokens)

        # Start with the initial prompt.
        print(prompt, end="", flush=True)
    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    end_loop = False
    while not end_loop:
        response = await model.next_token(request_id_context_dict)
        if len(response) == 0:
            break
        for key, response_text in response.items():
            if key not in response:
                end_loop = True
                break
            if metrics:
                if is_first_token[key]:
                    is_first_token[key] = False
                    # TODO(MSDK-973): This captures first tokens for all <batch size>
                    # prompts. Not sure if it messes up our metrics though.
                    metrics.signpost("first_token")
                metrics.new_token()
            # TODO(MSDK-974): Clean up console prints (row vs. col order) when batch_size > 1.
            print(response_text, end="", flush=True)
    if metrics:
        metrics.signpost("end_generation")
    print()
