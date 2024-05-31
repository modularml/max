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


def get_next_token(inout s: StringRef) -> (StringRef, TokenType):
    """Gets the next token within the limits and returns the unscanned indices.

    Args:
        s: JSON string, which is advanced beyond consumed bytes.
    Returns:
        Tuple of (
            Substring containing the token contents
            The type of token returned.
        )

    """

    # Skip the white spaces.
    while True:
        if s.empty():
            return StringRef(), TokenType.end

        if s[0] in WS:
            s = s.drop_front()
        else:
            break

    # Keep track of how many bytes are in this token.
    var token = s.take_front(1)
    var i = 1
    var end_idx = len(s)
    var token_type: TokenType

    # TODO: Why doesn't StringRef have a normal getitem?
    var c = String(s[0])

    # Detect which type of token this is.
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
        while True:
            if i == end_idx:
                raise "Could not find end double quotes."

            if s[i] == ESCAPE:
                if i + 1 == end_idx:
                    raise "escape at end of line."
                i += 1  # Skip the next character
            elif s[i] == DOUBLE_QUOTE:
                break
            i += 1

        # Crop the double quotes from the token.
        token = s.drop_front(1).take_front(i - 1)
        token_type = TokenType.string

        # Move the i one more char, since it's a double-quote that's part of
        # this string.
        i += 1
    elif c in INITIAL_NUMBER_CHARS:
        while i < end_idx:
            if s[i] not in NUMBER_CHARS:
                break
            i += 1
        # TODO: Validate number
        token = s.take_front(i)
        token_type = TokenType.number
    elif islower(ord(c)):
        # Check if the next token is "true", "false" or "null"
        var first_idx = i
        while i < end_idx and islower(ord(s[i])):
            i += 1

        token = s.take_front(i)
        if token == NULL:
            token_type = TokenType.null
        elif token == TRUE:
            token_type = TokenType.bool
        elif token == FALSE:
            token_type = TokenType.bool
        else:
            raise 'Invalid token "' + str(token) + '" in "' + String(s) + '"'

    else:
        var start = max(0, i - 20)
        var end = min(end_idx, i + 20)
        raise (
            "Unable to parse token: "
            + c
            + " (ord="
            + str(ord(c))
            + ")\n"
            + "Context: "
            + String(s)[start:end]
        )
    s = s.drop_front(i)
    return token, token_type


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

    @staticmethod
    def from_string(s: StringRef) -> Self:
        js = _from_string(s)

        # Make sure nothing appears afterwards:
        token, token_type = get_next_token(s)
        if token_type != TokenType.end:
            raise "Unexpected token found: " + str(token)

        return js

    def get(self, args: List[String]) -> Node:
        var node = self.root
        if len(args) == 0:
            return node
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

    def get(self, *args: String) -> Node:
        # Convert to list -- can't do self.get(args) :(
        var args_list = List[String]()
        for ele in args:
            args_list.append(ele[])
        return self.get(args_list)


def _from_string(inout s: StringRef) -> JsonStorage:
    # Dict and Arrays will want the entire span as their location.
    orig_buffer = s

    token, token_type = get_next_token(s)
    root = Node(token_type.to_node_type(), token, -1)

    var storage = List[Dict[String, Node]]()
    if token_type == TokenType.object_open:
        root.storage_index = 0
        var root_storage = Dict[String, Node]()
        storage.append(root_storage)

        var object_closed = False
        while not object_closed:
            # Look ahead to see if the next token is a "}"
            var temp = s
            token, token_type = get_next_token(temp)
            if token_type == TokenType.object_close:
                object_closed = True
                s = temp
                break

            # Read the next token (object key)
            token, token_type = get_next_token(s)
            var key = token
            if token_type == TokenType.end:
                break
            elif token_type == TokenType.object_close:
                object_closed = True
                break
            elif token_type != TokenType.string:
                raise "JSON key must be a string, got: " + str(token)

            # Consume the next token (should be a ':')
            token, token_type = get_next_token(s)
            if token_type != TokenType.colon:
                raise "Expected a ':' after string key, got: " + str(token)

            # Get the value using a recursive call to _from_string:
            var value = _from_string(s)

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
            token, token_type = get_next_token(s)
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
            root.value = orig_buffer
        else:
            raise "Invalid formatted JSON object. Object was never closed."
    elif token_type == TokenType.array_open:
        root.storage_index = 0
        var root_storage = Dict[String, Node]()
        storage.append(root_storage)

        # Arrays are also stored in a dict. The keys are the str(int index)
        # of each element.
        var key_int = 0

        var array_closed = False
        while True:
            # Look ahead to see if the next token is a "]"
            var temp = s
            token, token_type = get_next_token(temp)
            if token_type == TokenType.array_close:
                array_closed = True
                s = temp
                break

            # Get the value using a recursive call to _from_string:
            var value = _from_string(s)

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
            token, token_type = get_next_token(s)
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
            root.value = orig_buffer
        else:
            raise "Invalid formatted JSON array. Object was never closed."

    return JsonStorage(root, storage)
