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
import testing

from pipelines.replit.bpe_tokenizer.bpe_tokenizer import Tokenizer


def test_tokenizer():
    var s = "hello world"
    # This contains a subset of vocab and merges that could be used in
    # tokenizing "hello world".
    # The order of the merges matter since it's used to determine which
    # merge to do.
    var tokenizer_json: String = """{
        "model": {
            "vocab": {
                "d": 0,
                "e": 1,
                "h": 2,
                "l": 3,
                "o": 4,
                "r": 5,
                "w": 6,
                "Ġ": 7,
                "or": 8,
                "he": 9,
                "lo": 10,
                "el": 11,
                "Ġw": 12,
                "ll": 13,
                "ell": 14,
                "ld": 15,
                "rl": 16,
                "wo": 17,
                "orld": 18,
                "hel": 19,
                "hell": 20,
                "Ġworld": 21,
                "world": 22,
                "hello": 23,
                "wor": 24,
                "Ġwor": 25,
                },
            "merges": [
                "o r",
                "h e",
                "l o",
                "e l",
                "Ġ w",
                "l l",
                "el l",
                "l d",
                "r l",
                "w o",
                "or ld",
                "he l",
                "he ll",
                "Ġw orld",
                "w orld",
                "hel lo",
                "w or",
                "Ġw or"]
        }
    }
    """

    var tokenizer = Tokenizer.from_string(tokenizer_json^)
    tokens = tokenizer.encode(s)
    testing.assert_equal(2, len(tokens))
    testing.assert_equal("hello", tokens[0].token)
    testing.assert_equal(23, tokens[0].id)
    testing.assert_equal("Ġworld", tokens[1].token)
    testing.assert_equal(21, tokens[1].id)


def main():
    test_tokenizer()
