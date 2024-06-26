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

from pipelines.replit.bpe_tokenizer.json import JsonStorage, NodeType


def test_value():
    var js = JsonStorage.from_string("3453.23e-5")
    testing.assert_equal(NodeType.number, js.get().type)
    # TODO: Test cast to float (not currently supported)

    js = JsonStorage.from_string("-3453")
    testing.assert_equal(NodeType.number, js.get().type)
    testing.assert_equal(-3453, js.get().to_int())
    _ = js^  # Ensure that `js` is kept in memory

    js = JsonStorage.from_string('"regular string"')
    testing.assert_equal(NodeType.string, js.get().type)
    testing.assert_equal("regular string", str(js.get().value))
    _ = js^  # Ensure that `js` is kept in memory

    js = JsonStorage.from_string('"a double quote is \\""')
    testing.assert_equal(NodeType.string, js.get().type)
    testing.assert_equal('a double quote is \\"', str(js.get().value))
    _ = js^  # Ensure that `js` is kept in memory

    js = JsonStorage.from_string("true")
    testing.assert_equal(NodeType.bool, js.get().type)
    testing.assert_equal(True, js.get().to_bool())
    _ = js^  # Ensure that `js` is kept in memory

    js = JsonStorage.from_string("false")
    testing.assert_equal(NodeType.bool, js.get().type)
    testing.assert_equal(False, js.get().to_bool())
    _ = js^  # Ensure that `js` is kept in memory

    js = JsonStorage.from_string("null")
    testing.assert_equal(NodeType.null, js.get().type)
    testing.assert_equal("null", js.get().value)
    _ = js^  # Ensure that `js` is kept in memory


def test_invalid_value():
    with testing.assert_raises():
        var js = JsonStorage.from_string("abc")
    with testing.assert_raises():
        var js = JsonStorage.from_string("123 abc")
    with testing.assert_raises():
        var js = JsonStorage.from_string("3453abc")
    with testing.assert_raises():
        var js = JsonStorage.from_string("FALSE")


def test_empty_array():
    var js = JsonStorage.from_string("[]")
    testing.assert_equal(NodeType.array, js.get().type)
    testing.assert_equal(0, len(js.storage[0]))

    js = JsonStorage.from_string("  \n\t[   ]  ")
    testing.assert_equal(NodeType.array, js.get().type)
    testing.assert_equal(0, len(js.storage[0]))


def test_invalid_array():
    with testing.assert_raises():
        js = JsonStorage.from_string("[,]")
    with testing.assert_raises():
        js = JsonStorage.from_string("[}]")
    with testing.assert_raises():
        js = JsonStorage.from_string("[")
    with testing.assert_raises():
        js = JsonStorage.from_string("[[]")
    with testing.assert_raises():
        js = JsonStorage.from_string("[]]")


def test_simple_array():
    js = JsonStorage.from_string("[23, 53, true]")
    testing.assert_equal(NodeType.array, js.get().type)
    storage = js.storage[js.get().storage_index]
    testing.assert_equal(23, js.get("0").to_int())
    testing.assert_equal(53, js.get("1").to_int())
    testing.assert_equal(True, js.get("2").to_bool())
    testing.assert_equal(3, len(storage))
    with testing.assert_raises():
        _ = js.get("3")


def test_nested_array():
    js = JsonStorage.from_string(
        '[51, [32, 14], "Value", null, true, false, {}, {"key": "another"}]'
    )
    testing.assert_equal(NodeType.array, js.get().type)
    var storage = js.storage[js.get().storage_index]
    testing.assert_equal(8, len(storage))

    testing.assert_equal(51, js.get("0").to_int())

    with testing.assert_raises():
        testing.assert_equal(1, js.get("1").to_int())
    testing.assert_equal(NodeType.array, js.get("1").type)
    testing.assert_equal(32, js.get("1", "0").to_int())
    testing.assert_equal(14, js.get(List[String]("1", "1")).to_int())

    testing.assert_equal(NodeType.string, js.get("2").type)
    testing.assert_equal("Value", js.get("2").value)

    testing.assert_equal(NodeType.null, js.get("3").type)

    testing.assert_equal(True, js.get("4").to_bool())
    testing.assert_equal(False, js.get("5").to_bool())

    testing.assert_equal(NodeType.object, js.get("6").type)

    testing.assert_equal(NodeType.object, js.get("7").type)
    testing.assert_equal(NodeType.string, js.get("7", "key").type)
    testing.assert_equal("another", js.get("7", "key").value)
    _ = js^


def test_empty_object():
    var js = JsonStorage.from_string("{}")
    testing.assert_equal(NodeType.object, js.get().type)
    testing.assert_equal(0, len(js.storage[0]))
    _ = js^  # Ensure that `js` is kept in memory

    js = JsonStorage.from_string("  \n\t{  } ")
    testing.assert_equal(NodeType.object, js.get().type)
    testing.assert_equal(0, len(js.storage[0]))
    _ = js^  # Ensure that `js` is kept in memory


def test_invalid_object():
    with testing.assert_raises():
        js = JsonStorage.from_string("{:}")
    with testing.assert_raises():
        js = JsonStorage.from_string('{"key"}')
    with testing.assert_raises():
        js = JsonStorage.from_string('{"key":"key" "key"}')
    with testing.assert_raises():
        js = JsonStorage.from_string("{1: 2}")  # Keys can only be string
    with testing.assert_raises():
        js = JsonStorage.from_string('{"key":"key", "key"}')
    with testing.assert_raises():
        js = JsonStorage.from_string("{{}")


def test_simple_object():
    js = JsonStorage.from_string('{"key 1": 123, "key 2": false, "key 3": "5"}')
    testing.assert_equal(NodeType.object, js.get().type)
    testing.assert_equal(123, js.get("key 1").to_int())
    testing.assert_equal(False, js.get("key 2").to_bool())
    testing.assert_equal("5", js.get("key 3").value)
    with testing.assert_raises():
        _ = js.get("not key")
    storage = js.storage[js.get().storage_index]
    testing.assert_equal(3, len(storage))

    _ = js^  # Ensure that `js` is kept in memory


def test_nested_object():
    js = JsonStorage.from_string(
        '{"key 1": [1, "2", ["3"]], "key 2": {"key 1": 5.6},}'
    )
    testing.assert_equal(NodeType.object, js.get().type)

    testing.assert_equal(1, js.get("key 1", "0").to_int())

    with testing.assert_raises():
        _ = js.get("key 1", "1").to_int()
    testing.assert_equal("2", js.get("key 1", "1").value)

    testing.assert_equal(NodeType.number, js.get("key 2", "key 1").type)
    testing.assert_equal("5.6", js.get("key 2", "key 1").value)

    _ = js^  # Ensure that `js` is kept in memory


def main():
    test_value()
    test_invalid_value()
    test_empty_array()
    test_invalid_array()
    test_simple_array()
    test_nested_array()
    test_empty_object()
    test_invalid_object()
    test_simple_object()
    test_nested_object()
