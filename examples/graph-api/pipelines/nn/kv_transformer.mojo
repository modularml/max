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
from collections import Optional

from max.graph import ops, Symbol, _OpaqueType, TensorType, Type, Dim
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from max.tensor import Tensor, TensorShape


from .kv_attention import KVCacheOptimizedAttention
from pipelines.nn import Embedding, Linear, RMSNorm, FeedForward
from max.graph.kv_cache.types import (
    ContiguousKVCacheCollection,
    ContiguousKVCache,
    KVCacheStaticParams,
)
from max.graph.kv_cache.kernel_names import _kv_cache_kernel_names


def kv_cache_length[
    type: DType, kv_params: KVCacheStaticParams
](cache: Symbol) -> Symbol:
    """Retrieve the length of our ContiguousKVCache from mo.opaque type.

    Args:
        cache: KVCacheCollection mo.opaque type.

    Returns:
        Symbol: scalar with the length of the ContiguousKVCache.
    """
    alias kernel_names = _kv_cache_kernel_names[type, kv_params]()
    out_type = TensorType(DType.uint32)
    out = ops.custom[kernel_names.kv_cache_length_kernel](
        List[Symbol](cache), out_type
    )
    return out


@value
struct KVCacheOptimizedTransformerBlock[
    type: DType, kv_params: KVCacheStaticParams
](CollectionElement):
    """Transformer layer with our custom ContiguousKVCache mo.opaque type."""

    # Ops
    var attention: KVCacheOptimizedAttention[type, kv_params]
    var feed_forward: FeedForward
    var attention_norm: RMSNorm
    var ffn_norm: RMSNorm

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        freqs_cis: Symbol,
        kv_collection: Symbol,
        mask: Symbol,
        valid_lengths: Symbol,
    ) -> Symbol:
        """Constructs the forward pass over Transformer layer.
        Args:
            input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
            start_pos: Scalar with index of starting token, effectively tracks
                the number of entries in the cache.
            freqs_cis: Positional frequencies tensor with shape
                (seq_len, head_dim // 2, 2).
            kv_collection: The KVCacheCollection object containing our KVCache entries for each layer
            valid_lengths: The unpadded lengths of each sequence in the batch.

        Returns:
            Symbol: representing hidden_state with shape:
                (batch_size, seq_len, num_heads * head_dim)
        """

        # do attention
        attention_out = self.attention(
            self.attention_norm(input),
            start_pos,
            freqs_cis,
            kv_collection,
            mask,
            valid_lengths,
        )[0]

        # add residual
        h = input + attention_out

        # do FFN and add residual
        h = h + self.feed_forward(self.ffn_norm(h))
        return h


@value
struct KVCacheOptimizedTransformer[type: DType, kv_params: KVCacheStaticParams]:
    """Top-level block for Transformer-based models with custom ContiguousKVCache mo.opaque type.
    """

    alias max_seq_len = 2048

    # hyperparams
    var dim: Int
    var n_heads: Int

    # ops
    var embedding: Embedding
    var layers: List[KVCacheOptimizedTransformerBlock[type, kv_params]]
    var norm: RMSNorm
    var output: Linear
    var theta: Float64
    var rope_scaling: Optional[Symbol]

    def freqs_cis(self, seq_len: Symbol, seq_len_dim: Dim) -> Symbol:
        """Constructs the RoPE positional embeddings."""
        g = seq_len.graph()
        n = self.dim // self.n_heads
        iota = g.range[DType.float32](0, n - 1, 2)
        if self.rope_scaling:
            iota = iota * self.rope_scaling.value()
        freqs = 1.0 / (self.theta ** (iota / n))
        t = g.range[DType.float32](0, Self.max_seq_len * 2.0, 1)
        freqs = t.reshape(-1, 1) * freqs.reshape(1, -1)

        var retval = ops.stack(List(ops.cos(freqs), ops.sin(freqs)), axis=-1)
        return ops.cast(retval, type)

    def __call__(
        self,
        tokens: Symbol,
        mask: Symbol,
        kv_collection: Symbol,
    ) -> (Symbol, Symbol):
        """Constructs our model graph
        Args:
            tokens: Token IDs tensor with type Int64 and shape:
                (batch_size, prompt_seq_len).
            kv_collection: our mo.opaque KVCacheCollection object storing KVCache
                entries for each layer. Has logical shape:
                (num_layers, 2, batch_size, cache_seq_len, num_heads, head_size)

        Returns:
            Tuple of Symbols containing:
              - hidden_state with shape (batch_size, seq_len, num_heads * head_size).
              - mo.opaque KVCacheCollection type.
        """
        g = tokens.graph()
        start_pos = ops.cast(
            kv_cache_length[type, kv_params](kv_collection), DType.int64
        )
        h = self.embedding(tokens)
        seq_len = ops.shape_of(tokens)[1]
        freqs_cis = self.freqs_cis(seq_len, tokens.shape()[1])

        valid_length = ops.cast(ops.shape_of(tokens)[1], DType.uint32)
        valid_lengths = valid_length.broadcast_to(tokens.shape()[0])

        for i in range(len(self.layers)):
            h = self.layers[i](
                h,
                start_pos,
                freqs_cis,
                kv_collection,
                mask,
                valid_lengths,
            )

        return self.norm(h) @ self.output, kv_collection
