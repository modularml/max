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

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ReplitContext:
    """The context for text generation using a Replit model."""

    prompt: str
    """Input prompt string prior to tokenization."""

    max_tokens: int
    """The maximum number of tokens to generate, including the prompt."""

    _cache_seq_id: int
    """Sequence id to tell the KV cache manager which cache block this owns."""

    next_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    """A (seq_len,) vector of the input tokens for this iteration."""

    tokens: list[int] = field(default_factory=list)
    """Tokens generated so far."""

    decoded: str = ""
    """Decoded text sequence from `self.tokens` above."""

    def append(self, token_ids: np.ndarray, decoded: str) -> None:
        """Appends to the generated tokens and decoded output."""
        assert len(token_ids.shape) == 1
        self.next_tokens = token_ids
        self.tokens.extend(token_ids)
        self.decoded += decoded

    @property
    def cache_seq_id(self) -> int:
        """Returns the seq id for the KVCacheManager, ensuring it is set."""
        return self._cache_seq_id

    def is_done(self, eos: int) -> bool:
        """Returns true if token gen for this context completed, else false."""
        return self.tokens[-1] == eos or len(self.tokens) > self.max_tokens

    @property
    def seq_len(self) -> int:
        """Current sequence length: num tokens input this iteration.
        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        return self.next_tokens.shape[-1]
