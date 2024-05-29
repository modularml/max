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
from pathlib import Path

from utils.variant import Variant

from .ball import Ball
from .json import JsonStorage, NULL, NodeType
from .max_heap import MaxHeap, OrderableElement


@value
struct TokenWithID(CollectionElement, Stringable):
    """A string token, along with its ID in the vocabulary (or 0 for unk)."""

    var token: String
    var id: Int

    fn __str__(self) -> String:
        return "[" + str(self.token) + "=" + str(self.id) + "]"


@value
struct StringPair(KeyElement):
    var left: String
    var right: String

    fn __eq__(self, other: Self) -> Bool:
        return (self.left == other.left) and (self.right == other.right)

    fn __ne__(self, other: Self) -> Bool:
        return (self.left != other.left) or (self.right != other.right)

    fn __moveinit__(inout self, owned existing: Self):
        self.left = existing.left^
        self.right = existing.right^

    fn __copyinit__(inout self, existing: Self):
        self.left = existing.left
        self.right = existing.right

    fn __hash__(self) -> Int:
        return hash(self.left) * 12345 + hash(self.right)


@value
struct MergeScore(CollectionElement):
    var rank: Int
    var id: Int  # Vocab ID

    fn __moveinit__(inout self, owned existing: Self):
        self.rank = existing.rank
        self.id = existing.id

    fn __copyinit__(inout self, existing: Self):
        self.rank = existing.rank
        self.id = existing.id


@value
struct MergeOption(OrderableElement):
    """Metadata for tracking possible BPE merges in a priority queue."""

    var left: Ball[String].ID
    var right: Ball[String].ID
    var score: Float32
    var checksum: Int

    fn __lt__(self, other: Self) -> Bool:
        return (self.score < other.score) or (
            self.score == other.score and self.left > other.left
        )


@value
struct Tokenizer:
    var vocab: Dict[String, Int]
    var merges: Dict[StringPair, MergeScore]
    var json_str: String

    @staticmethod
    def from_file(path: Path) -> Self:
        return Self.from_string(path.read_text())

    @staticmethod
    fn from_string(owned s: String) raises -> Self:
        # NOTE: JsonStorage takes unsafe StringRef's into S, and `merges`
        # maintains those StringRefs for the duration of the Tokenizer object.
        # We make sure to keep s alive as long as we need it to avoid dangling
        # pointers.
        var j = JsonStorage.from_string(StringRef(s.unsafe_uint8_ptr(), len(s)))

        # Just read the vocab and merges, assume the configuration
        # parameters are as expected (e.g. type=BPE, byte_fallback=False, etc).
        var vocab_node = j.get("model", "vocab")
        var vocab_storage = j.storage[vocab_node.storage_index]
        var vocab = Dict[String, Int]()
        for item in vocab_storage.items():
            vocab[item[].key] = item[].value.to_int()

        var merge_node = j.get("model", "merges")
        var merge_storage = j.storage[merge_node.storage_index]
        var num_merges = len(merge_storage)

        var merges = Dict[StringPair, MergeScore]()
        for n in range(num_merges):
            var merge = merge_storage[str(n)].value
            var split = str(merge).split(" ")
            if len(split) != 2:
                raise "Invalid merge: " + str(merge)
            var merged = split[0] + split[1]
            try:
                var vocab_id = vocab[merged]
                # Set the merge score to the negative index to prioritize
                # earlier merges.
                merges[StringPair(split[0], split[1])] = MergeScore(
                    -n, vocab_id
                )
            except:
                raise "Could not find '" + str(merged) + "' in tokenizer vocab."

        return Self(vocab, merges, s^)

    fn encode(
        self,
        str: String,
        bos: Optional[String] = None,
        eos: Optional[String] = None,
    ) raises -> List[TokenWithID]:
        """Encode a string according to the BPE algorithm.

        The BPE vocabulary is a set of scored strings. BPE starts by
        considering every character in the input string as its own token,
        and then greedily merges the highest scoring adjacent pair
        until no more adjacent token merges exist in the vocabulary.

        We implement the tokens as a linked list, with a priority queue
        of merge options. We execute the highest-scoring merge, adding
        new merge options to the priority queue if they exist in the vocabulary.
        We can't remove out-dated merge options from the priority queue, so
        instead we add a checksum to them, which is the length of the merge
        they're expecting. Linked list elements only stop existing or grow
        in length, so we can always safely recognize an outdated merge.
        """
        var output = List[TokenWithID]()
        if bos and bos.value()[] in self.vocab:
            output.append(TokenWithID(bos.value()[], self.vocab[bos.value()[]]))

        var merge_options = MaxHeap[MergeOption]()
        var tokens = Ball[String]()

        @parameter
        fn maybe_add_merge(left: tokens.ID, right: tokens.ID) raises:
            var pair = StringPair(tokens[left], tokens[right])
            if pair in self.merges:
                var merge = self.merges[pair]
                var score = merge.rank
                merge_options.push(MergeOption(left, right, score, merge.id))

        # Initialize the tokens linked-list and initial merges.
        var prev: Optional[Ball[String].ID] = None
        for i in range(len(str)):
            var id = tokens.append(str[i].replace(" ", "Ġ"))
            if prev:
                maybe_add_merge(prev.value()[], id)
            prev = id

        while merge_options:
            var merge = merge_options.pop()
            # Check whether the best merge is still valid
            if merge.left not in tokens or merge.right not in tokens:
                continue  # outdated merge option

            var pair = StringPair(tokens[merge.left], tokens[merge.right])
            if (
                pair not in self.merges
                or self.merges[pair].id != merge.checksum
            ):
                continue  # outdated merge option
            # Merge the right token into the left token, then
            # add any new valid merge options to the priority queue.
            var left = tokens.prev(merge.left)
            var right = tokens.next(merge.right)
            tokens[merge.left] = tokens[merge.left] + tokens[merge.right]
            tokens.remove(merge.right)
            if right:
                maybe_add_merge(merge.left, right.value()[])
            if left:
                maybe_add_merge(left.value()[], merge.left)

        # Loop through the final list and construct the token sequence.
        var node_id = tokens._head
        while node_id:
            var id = node_id.value()[]
            var token = tokens[id]
            output.append(TokenWithID(token, self._encode_token(token)))
            node_id = tokens.next(id)

        if eos and eos.value()[] in self.vocab:
            output.append(TokenWithID(eos.value()[], self.vocab[eos.value()[]]))

        return output

    fn _encode_token(self, token: String) raises -> Int:
        return self.vocab.find(token).or_else(0)
