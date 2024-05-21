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
"""A simple generic max-heap implementation."""

from collections import List


trait Orderable:
    """Types which have a total order defined on them."""

    fn __lt__(self, other: Self) -> Bool:
        pass


trait OrderableElement(Orderable, CollectionElement):
    """Orderable types which are also CollectionElements."""

    pass


struct MaxHeap[ElementType: OrderableElement](Sized, Boolable):
    """A max-heap of an Orderable collection type.

    A MaxHeap is a convenient data structure for implementing a priority queue.

    Usage:
    ```mojo
    pq = MaxHeap[...]()
    pq.push(initial)
    while pq:
        var top = pq.pop()
        if something: pq.push(another)
    ```
    """

    var heap: List[ElementType]
    var begin_idx: Int

    fn __init__(inout self):
        """Constructs an empty heap."""
        self.heap = List[ElementType]()
        self.begin_idx = 0

    fn __len__(self) -> Int:
        """Checks how many elements are in the heap.."""
        return len(self.heap) - self.begin_idx

    fn __bool__(self) -> Bool:
        """Checks whether the heap has any elements in it."""
        return len(self) != 0

    fn push(inout self, owned elem: ElementType):
        """Adds a value to the heap."""
        self.heap.append(elem^)
        self._bubble_up(len(self.heap) - 1)

    fn pop(inout self) -> ElementType:
        """Removes the top element from the heap and return it."""
        debug_assert(self, "heap is empty")
        self._sink_down(self.begin_idx)

        var top = self.heap[self.begin_idx]
        self.begin_idx += 1
        return top

    fn _swap(inout self, i1: Int, i2: Int):
        # TODO: Swap syntax doesn't support non-register-passable types
        var tmp = self.heap[i1]
        self.heap[i1] = self.heap[i2]
        self.heap[i2] = tmp

    fn _bubble_up(inout self, idx: Int):
        if idx == self.begin_idx:
            return

        var parent_idx = self._parent_idx(idx)
        var parent = self.heap[parent_idx]
        var current = self.heap[idx]
        if parent < current:
            self._swap(parent_idx, idx)
            self._bubble_up(parent_idx)

    fn _sink_down(inout self, idx: Int):
        var li = self._left_child_idx(idx)
        var ri = self._right_child_idx(idx)

        var target_idx = idx  # smaller of the two children, if we should sink down
        if li < len(self.heap) - 1 and self.heap[target_idx] < self.heap[li]:
            target_idx = li
        elif ri < len(self.heap) - 1 and self.heap[target_idx] < self.heap[ri]:
            target_idx = ri

        if target_idx != idx:
            self._swap(idx, target_idx)
            self._sink_down(target_idx)

    fn _real_idx(self, idx: Int) -> Int:
        return idx + self.begin_idx

    fn _parent_idx(self, idx: Int) -> Int:
        return (self._real_idx(idx) - 1) // 2

    fn _left_child_idx(self, idx: Int) -> Int:
        return self._real_idx(idx) * 2 + 1

    fn _right_child_idx(self, idx: Int) -> Int:
        return (self._real_idx(idx) * 2) + 2
