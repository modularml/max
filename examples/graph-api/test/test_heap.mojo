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

"""Unit tests for MaxHeap."""

from testing import *

from pipelines.tokenizer.max_heap import MaxHeap, MinHeap


def test_simple():
    heap = MaxHeap[Int]()
    for i in range(10):
        heap.push(i)

    for i in range(10):
        assert_equal(9 - i, heap.pop())


def test_min_heap_simple():
    heap = MinHeap[Int]()
    for i in range(10):
        heap.push(i)

    for i in range(10):
        assert_equal(i, heap.pop())


def main():
    test_simple()
    test_min_heap_simple()
