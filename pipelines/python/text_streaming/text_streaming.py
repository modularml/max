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
from collections import defaultdict
from typing import Optional

from max.pipelines.interfaces import (
    TokenGenerator,
    TokenGeneratorRequest,
    PipelineTokenizer,
)

from utils import TextGenerationMetrics

MODEL_NAME = "model"


async def stream_text_to_console(
    model: TokenGenerator,
    tokenizer: PipelineTokenizer,
    prompt: str,
    metrics: Optional[TextGenerationMetrics] = None,
    max_batch_size: int = 1,
    print_tokens: bool = True,
):
    # Length of request_id_context_dict should be == batch_size.
    request_id_context = {}

    # TODO(MSDK-972): Make this batch_size variable based on size of the request dict.
    # NOTE: This batch_size param also needs to be == config.max_cache_size of
    # the underlying pipeline config.
    batch_size = max_batch_size

    # create a dict of request_id: contexts
    decoded_responses = {}

    req_id = str(uuid.uuid4())
    context = await tokenizer.new_context(
        TokenGeneratorRequest(req_id, 0, prompt, MODEL_NAME)
    )
    decoded_responses[req_id] = [prompt]
    request_id_context[req_id] = context
    prompt_size = len(context.tokens)

    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    if print_tokens:
        print(prompt, end="", flush=True)

    first_token = True
    while True:
        responses = model.next_token(request_id_context)[0]
        if not responses:
            break

        for req_id, context in request_id_context.items():
            if req_id not in responses or context.is_done(tokenizer.eos):
                del request_id_context[req_id]
                continue

            encoded_text = responses[req_id]
            response_text = await tokenizer.decode(context, encoded_text)
            if metrics:
                if first_token:
                    first_token = False
                    metrics.signpost("first_token")
                metrics.new_token()
            if print_tokens:
                print(response_text, end="", flush=True)
            else:
                decoded_responses[req_id].append(response_text)

        if not request_id_context:
            break

    if metrics:
        metrics.signpost("end_generation")

    for context in request_id_context.values():
        model.release(context)

    if print_tokens:
        print()
