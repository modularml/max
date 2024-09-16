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
"""Interfaces for different pipeline behaviors."""

from typing import Generic, Optional, Protocol, TypeVar

Context = TypeVar("Context")


class TokenGenerator(Generic[Context], Protocol):
    """Interface for LLM token-generator models."""

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Context:
        ...

    async def next_token(
        self, batch: dict[str, Context]
    ) -> dict[str, str | None]:
        ...
