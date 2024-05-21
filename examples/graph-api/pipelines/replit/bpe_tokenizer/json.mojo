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

"""
Some helpful JSON utilities.

JsonStorage: A container for parsed JSON data.
Non-recursive, due to the
[bug with recursive structs](https://linear.app/modularml/issue/MOCO-577/%5Bmojo%5D-%5Bbug%5D-value-doesnt-work-on-recursive-structs).


```
var s: String = "{'version': 1, 'data': ['a', 'b', 'c']}"
var js = JsonStorage.from_string(s^)
```
All values can be accessed via `JsonStorage.get`

```
print(js.get()) #  Node(type=object, value="{...}")
print(js.get('version')) --> Node(type=number, value="1")
print(js.get('version').to_int()) --> 1
print(js.get('data')) --> Node(type=array, value="['a', 'b', 'c']")
print(js.get('data', '1')) --> Node(type=string, value="b")
```
"""

from collections import List, Dict, Set

var WS = Set[String](" ", "\n", "\t")
alias COLON = ":"
alias COMMA = ","
alias OBJECT_OPEN = "{"
alias OBJECT_CLOSE = "}"
alias ARRAY_OPEN = "["
alias ARRAY_CLOSE = "]"
alias DOUBLE_QUOTE = '"'
alias NULL = "null"
alias TRUE = "true"
alias FALSE = "false"
alias ESCAPE = "\\"
alias NULL_END = "\n"
var NUMBER_CHARS = Set[String](
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "E", "e", ".", "-"
)

# Create a separate set without the "e" so the `get_next_token` function can
# easily differentiate between a number and "true"/"false" literals (when
# searching from right-to-left, "true"/"false" start with "e")
var INITIAL_NUMBER_CHARS = Set[String](
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"
)


@value
struct NodeType(Stringable, EqualityComparable):
    alias object = NodeType(0)
    alias array = NodeType(1)
    alias string = NodeType(2)
    alias number = NodeType(3)
    alias bool = NodeType(4)
    alias null = NodeType(5)
    alias end = NodeType(6)

    var kind: UInt8

    fn __eq__(self, other: Self) -> Bool:
        return self.kind == other.kind

    fn __ne__(self, other: Self) -> Bool:
        return self.kind != other.kind

    fn __str__(self) -> String:
        if self.kind == 0:
            return "object"
        if self.kind == 1:
            return "array"
        if self.kind == 2:
            return "string"
        if self.kind == 3:
            return "number"
        if self.kind == 4:
            return "bool"
        if self.kind == 5:
            return "null"
        if self.kind == 6:
            return "end"
        return "unknown type"


@value
struct TokenType(EqualityComparable, Stringable, KeyElement):
    alias unknown = TokenType(0)
    alias object_open = TokenType(1)
    alias object_close = TokenType(2)
    alias array_open = TokenType(3)
    alias array_close = TokenType(4)
    alias colon = TokenType(5)
    alias comma = TokenType(6)
    alias string = TokenType(7)
    alias number = TokenType(8)
    alias bool = TokenType(9)
    alias null = TokenType(10)
    alias end = TokenType(11)

    var kind: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.kind == other.kind

    fn __ne__(self, other: Self) -> Bool:
        return self.kind != other.kind

    fn __str__(self) -> String:
        if self.kind == 0:
            return "unknown"
        if self.kind == 1:
            return "object_open"
        if self.kind == 2:
            return "object_close"
        if self.kind == 3:
            return "array_open"
        if self.kind == 4:
            return "array_close"
        if self.kind == 5:
            return "colon"
        if self.kind == 6:
            return "comma"
        if self.kind == 7:
            return "string"
        if self.kind == 8:
            return "number"
        if self.kind == 9:
            return "bool"
        if self.kind == 10:
            return "null"
        if self.kind == 11:
            return "end"
        return "unknown type"

    fn __hash__(self) -> Int:
        """Return a 64-bit hash of the type's data."""
        return self.kind

    fn to_node_type(self) raises -> NodeType:
        if self.kind == 1:
            return NodeType.object
        elif self.kind == 3:
            return NodeType.array
        elif self.kind == 7:
            return NodeType.string
        elif self.kind == 8:
            return NodeType.number
        elif self.kind == 9:
            return NodeType.bool
        elif self.kind == 10:
            return NodeType.null
        elif self.kind == 11:
            return NodeType.end
        raise "Cannot convert token type " + str(self) + " into a NodeType."


var VALUE_TYPES = Set[TokenType](
    TokenType.bool, TokenType.number, TokenType.null, TokenType.string
)


def get_next_token(s: String, start: Int = 0) -> (StringRef, TokenType, Int):
    """Gets the next token within the limits and returns the unscanned indices.

    Args:
        s: JSON string.
        start: Index of `s` to start scanning. Must be `>=0` and `<len(s)`.
    Returns:
        Tuple of (
            Substring containing the token contents
            The type of token returned.
            Index of next unscanned character in `s`.
        )

    """
    var i = start
    var end_idx = len(s) - 1

    # Skip the white spaces.
    while i <= end_idx:
        if s[i] in WS:
            i += 1
        else:
            break
    if i > end_idx:
        return StringRef(s.unsafe_uint8_ptr(), 0), TokenType.end, i

    # Detect which type of token this is.
    var token = StringRef(s.unsafe_uint8_ptr() + i, 1)
    var token_type = TokenType.unknown

    var c = s[i]
    i += 1
    if c == OBJECT_OPEN:
        token_type = TokenType.object_open
    elif c == OBJECT_CLOSE:
        token_type = TokenType.object_close
    elif c == ARRAY_OPEN:
        token_type = TokenType.array_open
    elif c == ARRAY_CLOSE:
        token_type = TokenType.array_close
    elif c == COLON:
        token_type = TokenType.colon
    elif c == COMMA:
        token_type = TokenType.comma
    elif c == DOUBLE_QUOTE:
        var first_idx = i
        while i <= end_idx:
            if s[i] == ESCAPE:
                i += 1  # Skip the next character
            elif s[i] == DOUBLE_QUOTE:
                break
            i += 1
        if i > end_idx:
            raise "Could not find end double quotes."
        var length = abs(first_idx - i)
        # Crop the double quotes from the token.
        token = StringRef(s.unsafe_uint8_ptr() + first_idx, length)
        token_type = TokenType.string

        # Move the i one more char, since it's a double-quote that's part of
        # this string.
        i += 1
    elif c in INITIAL_NUMBER_CHARS:
        var first_idx = i
        while i <= end_idx:
            if s[i] not in NUMBER_CHARS:
                break
            i += 1
        # TODO: Validate number
        var length = abs(first_idx - i) + 1
        token = StringRef(s.unsafe_uint8_ptr() + first_idx - 1, length)
        token_type = TokenType.number
    elif islower(ord(c)):
        # Check if the next token is "true", "false" or "null"
        var first_idx = i
        while i <= end_idx:
            if not islower(ord(s[i])):
                break
            i += 1

        var length = abs(first_idx - i) + 1
        token = StringRef(s.unsafe_uint8_ptr() + first_idx - 1, length)
        if token == NULL:
            token_type = TokenType.null
        elif token == TRUE:
            token_type = TokenType.bool
        elif token == FALSE:
            token_type = TokenType.bool
        else:
            raise 'Invalid token "' + str(token) + '" at character ' + str(
                first_idx
            ) + " in " + s[first_idx:] + "\n\nFull string: " + s

    else:
        start = max(0, i - 20)
        end = min(end_idx, i + 20)
        raise (
            "Unable to parse token: "
            + c
            + " (ord="
            + str(ord(c))
            + ")\n"
            + "Context: "
            + s[start:end]
        )
    return token, token_type, i


@value
struct Node(Stringable):
    var type: NodeType
    var value: StringRef
    # Index into the parent JSON storage. Only valid for "Object" and "array"
    # node types.
    var storage_index: Int

    fn __str__(self) -> String:
        return (
            "Node(type="
            + str(self.type)
            + ", value="
            + str(self.value)
            + ", idx="
            + str(self.storage_index)
            + ")"
        )

    # TODO: Add `to_float`
    fn to_int(self) raises -> Int:
        if self.type == NodeType.number:
            return int(self.value)
        else:
            raise "Cannot convert node of type " + str(
                self.type
            ) + " to number."

    fn to_bool(self) raises -> Bool:
        if self.type == NodeType.bool:
            if self.value == TRUE:
                return True
            elif self.value == FALSE:
                return False
            else:
                raise "Something went wrong."
        else:
            raise "Cannot convert node of type " + str(self.type) + " to bool."


@value
struct JsonStorage:
    var root: Node
    var storage: List[Dict[String, Node]]
    var s: String

    @staticmethod
    fn from_string(owned s: String) raises -> Self:
        var js: Self
        var start: Int
        js, start = _from_string(s, 0)

        # Make sure nothing appears afterwards:
        var token: StringRef
        var token_type: TokenType

        token, token_type, start = get_next_token(s, start)
        if token_type != TokenType.end:
            raise "Unexpected token found: " + str(token)
        return JsonStorage(js.root, js.storage, s)

    fn get(self, args: List[String]) raises -> Node:
        if len(args) == 0:
            return self.root
        var node = self.root
        for n in range(len(args)):
            var key = args[n]
            if node.type != NodeType.object and node.type != NodeType.array:
                raise "Can't access key '" + key + "' from " + str(
                    node
                ) + " because it's not an array or object."

            try:
                node = self.storage[node.storage_index][key]
            except e:
                raise "Unable to get key '" + key + "' from " + str(node)
        return node

    fn get(self, *args: String) raises -> Node:
        # Convert to list -- can't do self.get(args) :(
        var args_list = List[String]()
        for ele in args:
            args_list.append(ele[])
        return self.get(args_list)


def _from_string(s: String, start: Int) -> (JsonStorage, Int):
    var token: StringRef
    var token_type: TokenType

    token, token_type, start = get_next_token(s, start)
    var root: Node
    try:
        root = Node(token_type.to_node_type(), token, -1)
    except:
        raise "Invalid formatted JSON: " + s + "\n Read token type " + token_type + ", but expected '{', '[', true, false, null, a string or a number."

    var storage = List[Dict[String, Node]]()
    if token_type == TokenType.object_open:
        var root_start = start - 1
        root.storage_index = 0
        var root_storage = Dict[String, Node]()
        storage.append(root_storage)

        var object_closed = False
        while not object_closed:
            # Look ahead to see if the next token is a "}"
            var temp: Int
            token, token_type, temp = get_next_token(s, start)
            if token_type == TokenType.object_close:
                object_closed = True
                start = temp
                break

            # Read the next token (object key)
            token, token_type, start = get_next_token(s, start)
            var key = token
            if token_type == TokenType.end:
                break
            elif token_type == TokenType.object_close:
                object_closed = True
                break
            elif token_type != TokenType.string:
                raise "JSON key must be a string, got: " + str(token)

            # Consume the next token (should be a ':')
            token, token_type, start = get_next_token(s, start)
            if token_type != TokenType.colon:
                raise "Expected a ':' after string key, got: " + str(token)

            # Get the value using a recursive call to _from_string:
            var value: JsonStorage
            value, start = _from_string(s, start)

            # Get the current length of `storage` which will be used to
            # increment all indices in the returned nodes.
            var inc = len(storage)
            value.root.storage_index += inc
            for d in value.storage:
                for ele in d[].values():
                    ele[].storage_index += inc

            storage.extend(value.storage)
            # Add the returned value storage to the root's storage.
            storage[0][key] = value.root  # root_storage[key] doesn't work

            # Consume the next token, which could end the object or should be
            # a comma.
            token, token_type, start = get_next_token(s, start)
            if token_type == TokenType.object_close:
                object_closed = True
                break
            elif token_type == TokenType.end:
                break
            elif token_type != TokenType.comma:
                raise "Invalid formatted JSON object. Expected a comma but got: " + str(
                    token
                )
        if object_closed:
            root.value = StringRef(
                s.unsafe_uint8_ptr() + root_start, start - root_start
            )
        else:
            raise "Invalid formatted JSON object. Object was never closed."
    elif token_type == TokenType.array_open:
        var root_start = start - 1
        root.storage_index = 0
        var root_storage = Dict[String, Node]()
        storage.append(root_storage)

        # Arrays are also stored in a dict. The keys are the str(int index)
        # of each element.
        var key_int = 0

        var array_closed = False
        while True:
            # Look ahead to see if the next token is a "]"
            var temp: Int
            token, token_type, temp = get_next_token(s, start)
            if token_type == TokenType.array_close:
                array_closed = True
                start = temp
                break

            # Get the value using a recursive call to _from_string:
            var value: JsonStorage
            value, start = _from_string(s, start)

            # Get the current length of `storage` which will be used to
            # increment all indices in the returned nodes.
            var inc = len(storage)
            value.root.storage_index += inc
            for d in value.storage:
                for ele in d[].values():
                    ele[].storage_index += inc
            storage.extend(value.storage)

            # Add the returned value storage to the root's storage.
            # (root_storage[key] doesn't work)
            storage[0][str(key_int)] = value.root
            key_int += 1

            # Consume the next token, which could end the array or should be
            # a comma.
            token, token_type, start = get_next_token(s, start)
            if token_type == TokenType.array_close:
                array_closed = True
                break
            elif token_type == TokenType.end:
                break
            elif token_type != TokenType.comma:
                raise "Invalid formatted JSON array. Expected a comma but got: " + str(
                    token
                )
        if array_closed:
            root.value = StringRef(
                s.unsafe_uint8_ptr() + root_start, start - root_start
            )
        else:
            raise "Invalid formatted JSON array. Object was never closed."

    return JsonStorage(root, storage, ""), start
