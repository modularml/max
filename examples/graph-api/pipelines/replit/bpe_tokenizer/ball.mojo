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
"""A simple arena linked-list implementation."""

from collections import List, Optional


@value
struct Node[T: CollectionElement](CollectionElement):
    """A node in the linked list."""

    var value: T
    var prev: Optional[Ball[T].ID]
    var next: Optional[Ball[T].ID]


struct Ball[T: CollectionElement]:
    """A doubly-linked-list with nodes in a memory arena.

    - Elements in the list have an ID which can be used to reference them again.
    - IDs will never change or be re-used. If an item is removed its ID is invalid.
    - Linked-list ops are done on the arena directly

    ```mojo
    from tokenizer.ball import Ball
    var list = Ball[Int]()
    var id1 = list.append(0)
    var id2 = list.append(1)
    list[id2] == 1
    list.next(id1).value()[] == id2
    list.prev(id2).value()[] == id1
    list.remove(id1)
    list._head.value()[] == id2
    (id1 in list) == False
    list[id2] = 3
    ```
    """

    alias ID = Int

    var _arena: List[Optional[Node[T]]]
    var _head: Optional[Self.ID]
    var _tail: Optional[Self.ID]

    fn __init__(inout self):
        """Constructs a new empty linked list."""
        self._arena = List[Optional[Node[T]]]()
        self._head = None
        self._tail = None

    fn __contains__(self, id: Self.ID) -> Bool:
        """Checks whether the node is still in the list."""
        return 0 <= id < len(self._arena) and self._arena[id]

    fn append(inout self, owned value: T) -> Self.ID:
        """Adds a new element to the back of the list."""
        var id = len(self._arena)
        var node = Node[T](value^, self._tail, None)
        if self._tail:
            self._get_node(self._tail.value()[]).next = id
        else:
            self._head = id
        self._tail = id
        self._arena.append(node)
        return id

    fn remove(inout self, id: Self.ID):
        """Removes an element from the list."""
        var node = self._arena[id]._value_copy()
        self._arena[id] = None
        if node.prev:
            self._get_node(node.prev.value()[]).next = node.next
        if node.next:
            self._get_node(node.next.value()[]).prev = node.prev
        if self._head.value()[] == id:
            self._head = node.next
        if self._tail.value()[] == id:
            self._tail = node.prev

    fn next(self, id: Self.ID) -> Optional[Self.ID]:
        """Gets the next item in the list, if any."""
        return self._get_node(id).next

    fn prev(self, id: Self.ID) -> Optional[Self.ID]:
        """Gets the previous item in the list, if any."""
        return self._get_node(id).prev

    fn _get_node[
        mutability: Bool,
        lifetime: AnyLifetime[mutability].type,
    ](self: Reference[Self, mutability, lifetime], id: Self.ID) -> ref [
        lifetime
    ] Node[T]:
        return self[]._arena.__get_ref(id)[].value()[]

    fn __getitem__[
        mutability: Bool,
        lifetime: AnyLifetime[mutability].type,
    ](self: Reference[Self, mutability, lifetime], id: Self.ID) -> ref [
        lifetime
    ] T:
        """Gets a reference to a value in the list."""
        return self[]._get_node(id).value
