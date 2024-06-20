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

from collections import List
from testing import assert_equal

from pipelines.llama2.tokenizer.bpe import BPETokenizer


def create_dummy_model() -> BPETokenizer:
    model = BPETokenizer()

    model.add_token("ab", 0.0)
    model.add_token("cd", -0.1)
    model.add_token("abc", -0.2)
    model.add_token("a", -0.3)
    model.add_token("b", -0.4)
    model.add_token("c", -0.5)
    model.add_token("qr", -0.5)
    model.add_token("x", -0.6)
    model.add_token("d", -0.7)

    return model^


def test_tokenization():
    var model = create_dummy_model()

    var tokens = model.encode(String("xabcabaabcddxabcabaabcdd"))

    var expected = List[String]()
    expected.append("x")
    expected.append("abc")
    expected.append("ab")
    expected.append("a")
    expected.append("ab")
    expected.append("cd")
    expected.append("d")
    expected.append("x")
    expected.append("abc")
    expected.append("ab")
    expected.append("a")
    expected.append("ab")
    expected.append("cd")
    expected.append("d")

    assert_equal(len(tokens), len(expected))

    for i in range(len(tokens)):
        assert_equal(model.decode(tokens[i]), expected[i])


fn test_decode() raises:
    var model = create_dummy_model()

    var encoded = model.encode(String("qrcbaabccdabqrcbaabccdab"))
    var decoded = String()
    for token in encoded:
        decoded += model.decode(token[])

    assert_equal(decoded, "qrcbaabccdabqrcbaabccdab")


fn main() raises:
    test_tokenization()
    test_decode()
