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

from .config import InferenceConfig
from .context import ReplitContext
from typing import Optional
from max.pipelines import TokenGenerator


class Replit(TokenGenerator):
    """The overall interface to the Replit model."""

    def __init__(self, config: InferenceConfig):
        raise NotImplementedError("replit not yet implemented.")

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> ReplitContext:
        raise NotImplementedError("replit not yet implemented.")

    async def next_token(
        self, batch: dict[str, ReplitContext]
    ) -> dict[str, str]:
        raise NotImplementedError("replit not yet implemented.")

    async def release(self, context: ReplitContext):
        raise NotImplementedError("replit not yet implemented.")
