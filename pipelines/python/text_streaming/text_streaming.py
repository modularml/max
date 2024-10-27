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

from max.pipelines.interfaces import (
    TokenGenerator,
    TokenGeneratorRequest,
    TokenGeneratorTokenizer,
)

from utils import TextGenerationMetrics


async def stream_text_to_console(
    model: TokenGenerator,
    tokenizer: TokenGeneratorTokenizer,
    prompt: str,
    metrics: Optional[TextGenerationMetrics] = None,
    max_batch_size: int = 1,
    print_tokens: bool = True,
    n_duplicate: int = 1,
):
    if n_duplicate > max_batch_size:
        msg = (
            f"prompt is duplicated {n_duplicate} times, while"
            f" max batch size is {max_batch_size}"
        )
        raise ValueError(msg)

    # Length of request_id_context_dict should be == batch_size.
    request_id_context = dict()

    # TODO(MSDK-972): Make this batch_size variable based on size of the request dict.
    # NOTE: This batch_size param also needs to be == config.max_cache_size of
    # the underlying pipeline config.
    batch_size = max_batch_size
    # Special case UX to see response print as generated when batch_size == 1
    print_as_generated = batch_size == 1
    responses = {}

    # create a dict of request_id: contexts
    for _ in range(n_duplicate):
        # We make the key unique even for the same prompts for now.
        req_id = str(uuid.uuid4())
        context = await tokenizer.new_context(
            TokenGeneratorRequest("", 0, "", "")
        )
        responses[req_id] = [prompt]
        request_id_context[req_id] = context
        prompt_size = len(context.tokens)

    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    if print_tokens and print_as_generated:
        print(prompt, end="", flush=True)

    end_loop = False
    first_token = True
    while not end_loop:
        response = model.next_token(request_id_context)[0]
        if len(response) == 0:
            break
        for key, response_text in response.items():
            if key not in response:
                end_loop = True
                break
            if metrics:
                if first_token:
                    first_token = False
                    metrics.signpost("first_token")
                metrics.new_token()
            if print_tokens and print_as_generated:
                print(response_text, end="", flush=True)
            else:
                responses[key].append(response_text)
    if metrics:
        metrics.signpost("end_generation")

    for context in request_id_context.values():
        model.release(context)

    # Print prompt + response for each unique prompt
    if print_tokens and not print_as_generated:
        for response in responses.values():
            print("\n---\n")
            print("".join(response), flush=True)

    if print_tokens:
        print()
