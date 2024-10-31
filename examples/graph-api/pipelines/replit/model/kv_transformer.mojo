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

from max.graph import ops, Symbol, _OpaqueType, TensorType, Type, Dim, Graph
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from max.tensor import Tensor, TensorShape
import math
from utils.numerics import min_finite

from pipelines.nn import Linear, Embedding, FeedForward
from ..model.block import MPTMLP
from ..model.norm import LPLayerNorm
from ..model.kv_attention import KVCacheOptimizedAttention
from kv_cache.types import (
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


def gen_slopes(g: Graph, n_heads: Int32, alibi_bias_max: Int32 = 8) -> Symbol:
    """Generates slopes for ALIBI positional embedding"""
    ceil = math.ceil(math.log2(n_heads.cast[DType.float32]()))
    two_simd = Float32(2)
    _n_heads = pow(two_simd, ceil).cast[DType.int32]()
    m = ops.cast(g.range[DType.int32](1, _n_heads + 1, 1), DType.float32)
    m = m * g.scalar(
        alibi_bias_max.cast[DType.float32]() / _n_heads.cast[DType.float32]()
    )
    pow_ = ops.pow(g.scalar(Float32(2)), m)
    slopes = ops.div(g.scalar(Float32(1)), pow_)
    if _n_heads != n_heads:
        # TODO: Update to slopes[1::2] and slopes[::2] when slicing is fixed.
        slopes = ops.concat(
            List[Symbol](
                slopes[1 : int(_n_heads) : 2], slopes[0 : int(_n_heads) : 2]
            )
        )
        slopes = slopes[: int(n_heads)]
    return slopes.reshape(1, int(n_heads), 1, 1)


def attn_bias[
    type: DType
](
    g: Graph,
    mask: Symbol,
    n_heads: Int,
    seq_len: Int,
    alibi_bias_max: Int32,
) -> Symbol:
    """Generates an ALIBI attention bias"""
    with g.layer("_attn_bias"):
        alibi_bias = ops.cast(
            g.range[DType.int32](1 - seq_len, 1, 1),
            type,
        )
        alibi_bias = alibi_bias.reshape(1, 1, 1, seq_len)
        slopes = gen_slopes(g, n_heads, alibi_bias_max)
        attn_bias = ops.cast(alibi_bias * slopes, type)
        s_k = ops.shape_of(mask)[-1]
        out_dims = List[Dim](1, n_heads, 1, mask.shape()[-1])
        attn_bias = attn_bias[:, :, :, -s_k:, out_dims=out_dims]
        mask = mask.reshape(mask.shape()[0], 1, 1, mask.shape()[1])
        min_val = g.scalar(min_finite[type]())
        attn_bias = ops.select(mask, attn_bias, min_val)

        return attn_bias


@value
struct KVCacheOptimizedTransformerBlock[
    type: DType, kv_params: KVCacheStaticParams
](CollectionElement):
    """Transformer layer with our custom ContiguousKVCache mo.opaque type."""

    # Ops
    var attention: KVCacheOptimizedAttention[type, kv_params]
    var feed_forward: MPTMLP
    var attention_norm: LPLayerNorm[type]
    var ffn_norm: LPLayerNorm[type]

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        kv_collection: Symbol,
        attn_weight: Symbol,
        valid_lengths: Symbol,
    ) -> Symbol:
        """Constructs the forward pass over Transformer layer.
        Args:
            input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
            start_pos: Scalar with index of starting token, effectively tracks
                the number of entries in the cache.
            kv_collection: The KVCacheCollection object containing our KVCache entries for each layer
            attn_weight: The positional bias and casual mask for the current request.
            valid_lengths: The unpadded lengths of each sequence in the batch.
        Returns:
            Symbol: representing hidden_state with shape:
                (batch_size, seq_len, num_heads * head_dim)
        """

        # do attention
        attention_out = self.attention(
            self.attention_norm(input),
            start_pos,
            kv_collection,
            attn_weight,
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

    # hyperparams
    var dim: Int
    var n_heads: Int
    var alibi_bias_max: Int
    var max_seq_len: Int

    # ops
    var embedding: Embedding
    var layers: List[KVCacheOptimizedTransformerBlock[type, kv_params]]
    var norm: LPLayerNorm[type]
    var output: Linear
    var theta: Float64

    def __call__(
        self,
        tokens: Symbol,
        kv_collection: Symbol,
        attention_mask: Symbol,
    ) -> (Symbol, Symbol):
        """Constructs our model graph.
        Args:
            tokens: Token IDs tensor with type Int64 and shape:
                (batch_size, prompt_seq_len).
            kv_collection: our mo.opaque KVCacheCollection object storing KVCache
                entries for each layer. Has logical shape:
                (num_layers, 2, batch_size, cache_seq_len, num_heads, head_size)
            attention_mask: The attention mask for our given sequence
                with shape (batch_size, full_seq_len)

        Returns:
            Tuple of Symbols containing:
              - hidden_state with shape (batch_size, seq_len, num_heads * head_size).
              - mo.opaque KVCacheCollection type.
        """
        g = tokens.graph()

        # embed the tokens
        h = self.embedding(tokens)

        # create named dims and shape symbols
        full_seq_len = Dim("full_seq_len")
        seq_len = Dim("seq_len")
        seq_len_sym = ops.shape_of(tokens)[1]
        full_seq_len_sym = ops.shape_of(attention_mask)[1]
        start_pos = kv_cache_length[type, kv_params](kv_collection)

        # build our ALIBI biases
        attention_bias = attn_bias[type](
            g,
            attention_mask,
            self.n_heads,
            self.max_seq_len,
            self.alibi_bias_max,
        )

        # construct the causal mask
        causal_mask = ops.band_part(
            g.full[DType.bool](1, full_seq_len, full_seq_len),
            g.scalar[DType.int64](-1),
            num_upper=g.scalar[DType.int64](0),
            exclude=False,
        )

        causal_mask = causal_mask[
            -seq_len_sym:,
            -full_seq_len_sym:,
            out_dims = List[Dim](seq_len, full_seq_len),
        ].reshape(1, 1, seq_len, full_seq_len)

        # combine ALIBI biases and causal mask
        min_val = g.scalar(Scalar[type].MIN)
        attn_mask = ops.select(causal_mask, attention_bias, min_val)

        # construct valid lengths TODO callout need  to make this work for differing batch sizes
        valid_length = ops.cast(ops.shape_of(tokens)[1], DType.uint32)
        valid_lengths = valid_length.broadcast_to(tokens.shape()[0])

        for i in range(len(self.layers)):
            h = self.layers[i](
                h,
                start_pos,
                kv_collection,
                attn_mask,
                valid_lengths,
            )

        return self.output(self.norm(h)), kv_collection
