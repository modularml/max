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
"""TikToken implementation."""

from base64 import b64decode
from collections import Dict
from pathlib import Path
from .bpe import BPETokenizer, TokenWithID
from .regex import Match, Regex, CompileOption
from ...weights.gguf import GGUFArray, GGUFString


def _next_rune(inout span: Span[UInt8, _]) -> Int:
    if not span[0] & 0x80:
        result = int(span[0])
        span = span[1:]
        return int(result)
    elif not span[0] & 0x20:
        result = ((int(span[0]) & 0x1F) << 6) | (int(span[1]) & 0x3F)
        span = span[2:]
        return int(result)
    elif not span[0] & 0x10:
        result = (
            ((int(span[0]) & 0x0F) << 12)
            | ((int(span[1]) & 0x3F) << 6)
            | (int(span[2]) & 0x3F)
        )
        span = span[3:]
        return int(result)
    else:
        result = (
            ((int(span[0]) & 0x07) << 18)
            | ((int(span[1]) & 0x3F) << 12)
            | ((int(span[2]) & 0x3F) << 6)
            | (int(span[3]) & 0x3F)
        )
        span = span[4:]
        return int(result)


def _runes(string: String) -> List[Int]:
    span = string.as_bytes_slice()
    runes = List[Int]()
    while len(span):
        runes.append(_next_rune(span))
    return runes^


def _decode_token(string: String, decode_map: Dict[Int, UInt8]) -> String:
    result = List[UInt8]()
    for rune in _runes(string):
        result.append(decode_map[rune[]])
    result.append(0)
    return result


def _decode_map() -> Dict[Int, UInt8]:
    # I have no idea why this is the way it is.
    decode_map = Dict[Int, UInt8]()
    for i in range(256, 289):  # 0-32
        decode_map[i] = i - 256
    for i in range(33, 127):  # 33-126
        decode_map[i] = i
    for i in range(289, 323):  # 127-160
        decode_map[i] = i - 162
    for i in range(161, 256):  # 161-255
        decode_map[i] = i
    decode_map[323] = 173
    return decode_map^


struct TikTokenEncoder:
    var bpe: BPETokenizer
    var regex: Regex
    var special_tokens: Dict[String, Int]

    def __init__(
        inout self,
        owned bpe: BPETokenizer,
        owned regex: Regex,
        owned special_tokens: Dict[String, Int],
    ):
        self.bpe = bpe^
        self.regex = regex^
        self.special_tokens = special_tokens^

    @staticmethod
    def cl100k_base_llama3(path: Path) -> Self:
        return Self.cl100k_base_llama3(BPETokenizer.from_tiktoken(path))

    @staticmethod
    def cl100k_base_llama3(tokens: GGUFArray) -> Self:
        bpe = BPETokenizer()
        decode_map = _decode_map()
        for i in range(tokens.n):
            encoded = str(tokens.data[i])
            bpe.add_token(_decode_token(encoded, decode_map), i)
        return Self.cl100k_base_llama3(bpe)

    @staticmethod
    def cl100k_base_llama3(owned bpe: BPETokenizer) -> Self:
        special_tokens = Dict[String, Int]()
        special_tokens["<|begin_of_text|>"] = 128000
        special_tokens["<|end_of_text|>"] = 128001
        for e in special_tokens.items():
            bpe.add_token(e[].key, e[].value)

        pattern = str("|").join(
            "'[sdmt]|ll|ve|re",
            "[^\r\n[:alnum:]]?[[:alpha:]]+",
            "[[:digit:]]{1,3}",
            " ?[^[:space:][:alnum:]]+[\r\n]*",
            "[[:space:]]*[\r\n]",
            "([[:space:]]+)[[:space:]]",
            "[[:space:]]+",
        )

        return Self(bpe^, Regex(pattern, CompileOption.ICASE), special_tokens^)

    def encode(
        self,
        string: String,
        bos: Optional[String] = str("<|begin_of_text|>"),
        eos: Optional[String] = None,
    ) -> List[TokenWithID]:
        # Compared to Rust tiktoken, this does not currently implement
        # - special tokens  (not used in llama3)
        # - multithreaded decoding
        #   multithreaded decoding is quite a bit more complex, as it requires
        #   both splitting the text across threads and merging afterwards, and also
        #   needs to know how to handle the boundary conditions between different segments
        tokens = List[TokenWithID]()
        if bos:
            tokens += self.encode_special(bos.value()[])

        for segment in self.regex.findall(string, negative_lookahead_hack=True):
            ss = str(segment)
            if token_id := self.bpe.token_ids.find(ss):
                tokens += TokenWithID(ss^, token_id.value()[])
            else:
                tokens += self.bpe.encode(ss^)

        if eos:
            tokens += self.encode_special(eos.value()[])

        return tokens

    def encode_special(self, string: String) -> TokenWithID:
        if special_id := self.special_tokens.find(string):
            return TokenWithID(string, special_id.value()[])
        return TokenWithID(string, self.bpe.token_ids[string])

    def decode(self, token_id: Int) -> TokenWithID:
        return TokenWithID(self.bpe.vocab[token_id].token, token_id)
