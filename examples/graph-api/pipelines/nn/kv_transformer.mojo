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
from max.graph import ops, Symbol, _OpaqueType, TensorType, Type, Dim
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from max.tensor import Tensor, TensorShape


from .kv_attention import KVCacheOptimizedAttention
from pipelines.nn import Embedding, Linear, RMSNorm, FeedForward
from max.serve.kv_cache.types import (
    ContiguousKVCacheCollection,
    ContiguousKVCache,
    KVCacheLayout,
    KVCacheStaticParams,
)
from max.serve.kv_cache.kernel_names import _kv_cache_kernel_names


def kv_cache_length[kv_params: KVCacheStaticParams](cache: Symbol) -> Symbol:
    """Retrieve the length of our ContiguousKVCache from mo.opaque type.

    Args:
        cache: KVCacheCollection mo.opaque type.

    Returns:
        Symbol: scalar with the length of the ContiguousKVCache.
    """
    alias kernel_names = _kv_cache_kernel_names[DType.float32, kv_params]()
    out_type = TensorType(DType.int64)
    out = ops.custom[kernel_names.kv_cache_length_kernel](
        List[Symbol](cache), out_type
    )
    return out


def key_cache_for_layer[
    kv_params: KVCacheStaticParams
](cache: Symbol, layer_idx: Int) -> Symbol:
    """Retrieve the ContiguousKVCache object for the keys of layer at layer_idx.

    Args:
        cache: KVCacheCollection mo.opaque type.

    Returns:
        Symbol: mo.opaque ContiguousKVCache type corresponding to key_cache for layer_idx.
    """
    alias kernel_names = _kv_cache_kernel_names[DType.float32, kv_params]()

    g = cache.graph()

    layer_idx_tens = Tensor[DType.int64](
        TensorShape(
            1,
        )
    )
    layer_idx_tens[0] = layer_idx
    layer_idx_sym = g.constant(layer_idx_tens)

    out_type = _OpaqueType(ContiguousKVCache[DType.float32, kv_params].id())
    out = ops.custom[kernel_names.key_cache_for_layer_kernel](
        List[Symbol](layer_idx_sym, cache), out_type
    )
    return out


def value_cache_for_layer[
    kv_params: KVCacheStaticParams
](cache: Symbol, layer_idx: Int) -> Symbol:
    """Retrieve the ContiguousKVCache object for the keys of layer at layer_idx.

    Args:
        cache: KVCacheCollection mo.opaque type.

    Returns:
        Symbol: mo.opaque ContiguousKVCache type corresponding to value_cache for layer_idx.

    """
    alias kernel_names = _kv_cache_kernel_names[DType.float32, kv_params]()

    g = cache.graph()

    layer_idx_tens = Tensor[DType.int64](
        TensorShape(
            1,
        )
    )
    layer_idx_tens[0] = layer_idx
    layer_idx_sym = g.constant(layer_idx_tens)

    out_type = _OpaqueType(ContiguousKVCache[DType.float32, kv_params].id())
    out = ops.custom[kernel_names.value_cache_for_layer_kernel](
        List[Symbol](layer_idx_sym, cache), out_type
    )
    return out


@value
struct KVCacheOptimizedTransformerBlock[kv_params: KVCacheStaticParams](
    CollectionElement
):
    """Transformer layer with our custom ContiguousKVCache mo.opaque type."""

    # Ops
    var attention: KVCacheOptimizedAttention[kv_params]
    var feed_forward: FeedForward
    var attention_norm: RMSNorm
    var ffn_norm: RMSNorm

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        freqs_cis: Symbol,
        k_cache: Symbol,
        v_cache: Symbol,
        mask: Symbol,
    ) -> Symbol:
        """Constructs the forward pass over Transformer layer.
        Args:
            input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
            start_pos: Scalar with index of starting token, effectively tracks
                the number of entries in the cache.
            freqs_cis: Positional frequencies tensor with shape
                (seq_len, head_dim // 2, 2).
            k_cache: Previously computed keys. This is a mo.opaque ContiguousKVCache object
                with logical shape (batch, n_kv_heads, prev_seq_len, head_dim).
            v_cache: Previously computed values. This is a mo.opaque ContiguousKVCache object
                with logical shape (batch, n_kv_heads, prev_seq_len, head_dim).

        Returns:
            Symbol: representing hidden_state with shape:
                (batch_size, seq_len, num_heads * head_dim)
        """

        # do attention
        attention_out = self.attention(
            self.attention_norm(input),
            start_pos,
            freqs_cis,
            k_cache,
            v_cache,
            mask,
        )[0]

        # add residual
        h = input + attention_out

        # do FFN and add residual
        h = h + self.feed_forward(self.ffn_norm(h))
        return h


@value
struct KVCacheOptimizedTransformer[kv_params: KVCacheStaticParams]:
    """Top-level block for Transformer-based models with custom ContiguousKVCache mo.opaque type.
    """

    alias max_seq_len = 2048

    # hyperparams
    var dim: Int
    var n_heads: Int

    # ops
    var embedding: Embedding
    var layers: List[KVCacheOptimizedTransformerBlock[kv_params]]
    var norm: RMSNorm
    var output: Linear
    var theta: Float64

    def freqs_cis(
        self, start_pos: Symbol, seq_len: Symbol, seq_len_dim: Dim
    ) -> Symbol:
        """Constructs the RoPE positional embeddings."""
        g = start_pos.graph()
        n = self.dim // self.n_heads
        iota = g.range[DType.float32](0, n - 1, 2)
        freqs = 1.0 / (self.theta ** (iota / n))
        t = g.range[DType.float32](0, Self.max_seq_len * 2.0, 1)
        freqs = t.reshape(-1, 1) * freqs.reshape(1, -1)

        var retval = ops.stack(List(ops.cos(freqs), ops.sin(freqs)), axis=-1)
        return ops.cast(
            retval[
                start_pos : start_pos + seq_len, out_dims = List(seq_len_dim)
            ],
            DType.float32,
        )

    def __call__(
        self,
        tokens: Symbol,
        mask: Symbol,
        kv_cache: Symbol,
    ) -> (Symbol, Symbol):
        """Constructs our model graph
        Args:
            tokens: Token IDs tensor with type Int64 and shape:
                (batch_size, prompt_seq_len).
            kv_cache: our mo.opaque KVCacheCollection object storing KVCache
                entries for each layer. Has logical shape:
                (num_layers, 2, batch_size, cache_seq_len, num_heads, head_size)

        Returns:
            Tuple of Symbols containing:
              - hidden_state with shape (batch_size, seq_len, num_heads * head_size).
              - mo.opaque KVCacheCollection type.
        """
        g = tokens.graph()
        start_pos = kv_cache_length[kv_params](kv_cache)
        h = self.embedding(tokens)
        seq_len = ops.shape_of(tokens)[1]
        freqs_cis = self.freqs_cis(start_pos, seq_len, tokens.shape()[1])
        for i in range(len(self.layers)):
            k_cache = key_cache_for_layer[kv_params](kv_cache, i)
            v_cache = value_cache_for_layer[kv_params](kv_cache, i)
            h = self.layers[i](
                h,
                start_pos,
                freqs_cis,
                k_cache,
                v_cache,
                mask,
            )

        return self.norm(h) @ self.output, kv_cache
