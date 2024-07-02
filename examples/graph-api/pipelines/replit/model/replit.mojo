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
import math
from utils.numerics import min_finite
from collections import Optional, List

from max.graph import ops, Dim, TensorType, Symbol, Graph, Type
from max.tensor import TensorSpec
from max._driver import AnyTensor, Device

from ..layers.embedding import SharedEmbedding
from ..layers.block import MPTBlock
from ..layers.norm import LPLayerNorm
from ..weights.replit_checkpoint import Checkpoint
from ..weights.hyperparams import HyperParams


def gen_slopes(g: Graph, n_heads: Int32, alibi_bias_max: Int32 = 8) -> Symbol:
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


struct Replit[T: Checkpoint, dtype: DType]:
    """Replit model implementation.

    Parameters:
        T: Checkpoint used to load weights for this model.
        dtype: The DType of the weights and inputs to this model.
    """

    var hyperparams: HyperParams

    def __init__(inout self, hyperparams: HyperParams):
        self.hyperparams = hyperparams

    def create_empty_cache(self, device: Device) -> (AnyTensor, AnyTensor):
        head_dim = self.hyperparams.d_model // self.hyperparams.n_heads
        return (
            AnyTensor(
                device.allocate(
                    TensorSpec(
                        dtype,
                        self.hyperparams.num_blocks,
                        self.hyperparams.batch_size,
                        self.hyperparams.kv_n_heads,
                        head_dim,
                        0,
                    )
                )
            ),
            AnyTensor(
                device.allocate(
                    TensorSpec(
                        dtype,
                        self.hyperparams.num_blocks,
                        self.hyperparams.batch_size,
                        self.hyperparams.kv_n_heads,
                        0,
                        head_dim,
                    )
                ),
            ),
        )

    def _attn_bias(
        self, g: Graph, attention_mask: Optional[Symbol] = None
    ) -> Symbol:
        alibi_bias = ops.cast(
            g.range[DType.int32](1 - self.hyperparams.seq_len, 1, 1),
            dtype,
        )
        alibi_bias = alibi_bias.reshape(1, 1, 1, self.hyperparams.seq_len)
        slopes = gen_slopes(
            g, self.hyperparams.n_heads, self.hyperparams.alibi_bias_max
        )
        attn_bias = ops.cast(alibi_bias * slopes, dtype)
        if attention_mask:
            mask = attention_mask.value()
            s_k = ops.shape_of(mask)[-1]
            out_dims = List[Dim](
                1, self.hyperparams.n_heads, 1, mask.shape()[-1]
            )
            attn_bias = attn_bias[:, :, :, -s_k:, out_dims=out_dims]
            attn_bias_shape = ops.shape_of(attn_bias)
            broadcast_dims = List[Dim](
                1, self.hyperparams.n_heads, 1, mask.shape()[1]
            )
            mask = mask.reshape(mask.shape()[0], 1, 1, mask.shape()[1])
            mask = g.op(
                "mo.broadcast_to",
                List[Symbol](mask, attn_bias_shape),
                TensorType(mask.tensor_type().dtype, broadcast_dims),
            )
            min_val = g.op(
                "mo.broadcast_to",
                List[Symbol](g.scalar(min_finite[dtype]()), attn_bias_shape),
                TensorType(dtype, broadcast_dims),
            )
            attn_bias = ops.select(mask, attn_bias, min_val)
        return attn_bias

    def build_graph(
        self,
        name: String,
        params: T,
        with_attention_mask: Bool = False,
        use_cache: Bool = False,
    ) -> Graph:
        """Builds the replit model graph.

        The graph takes encoded tokens as input and outputs the predicted
        logits.

        Args:
            name: Name of the graph.
            params: Checkpoint class that loads parameter values.
            with_attention_mask: Whether to build the graph with an attention
              mask input.
            use_cache: Whether to build the graph with a key and value cache.
              When this is true, the updated cache values are included in the
              graph outputs.
        Returns:
            Replit Graph.
        """
        # Set up graph and inputs.
        seq_len = "seq_len"
        input_type = TensorType(
            DType.int64, self.hyperparams.batch_size, seq_len
        )
        in_types = List[Type](input_type)
        mask_input_idx = -1
        cache_input_idx = -1
        if with_attention_mask:
            attention_mask_type = TensorType(
                DType.bool, self.hyperparams.batch_size, "full_seq_len"
            )
            in_types.append(attention_mask_type)
            mask_input_idx = 1
            cache_input_idx = 2
        else:
            cache_input_idx = 1
        if use_cache:
            head_dim = self.hyperparams.d_model // self.hyperparams.n_heads
            prev_seq_len = "prev_seq_len"
            k_cache_type = TensorType(
                dtype,
                self.hyperparams.num_blocks,
                self.hyperparams.batch_size,
                self.hyperparams.kv_n_heads,
                head_dim,
                prev_seq_len,
            )
            v_cache_type = TensorType(
                dtype,
                self.hyperparams.num_blocks,
                self.hyperparams.batch_size,
                self.hyperparams.kv_n_heads,
                prev_seq_len,
                head_dim,
            )
            in_types.append(k_cache_type)
            in_types.append(v_cache_type)

        g = Graph(
            name,
            in_types=in_types,
        )

        @parameter
        def weight(name: String) -> Symbol:
            return g.constant(params.get[dtype](name))

        wte = SharedEmbedding(weight("transformer.wte.weight"))
        x = wte(g[0])
        if with_attention_mask:
            attn_bias = self._attn_bias(g, g[1])
        else:
            attn_bias = self._attn_bias(g)

        # Run through the transformer blocks. If the key and values are cached,
        # store the updated values which will later be concatenated and returned
        # as outputs.
        k_cache_updates = List[Symbol]()
        v_cache_updates = List[Symbol]()

        for i in range(self.hyperparams.num_blocks):
            block_prefix = "transformer.blocks." + str(i) + "."
            block = MPTBlock[dtype].create(
                params, block_prefix, g, self.hyperparams
            )
            if use_cache:
                k_cache = g[cache_input_idx][i]
                v_cache = g[cache_input_idx + 1][i]
                x, k_cache_update, v_cache_update = block(
                    x, attn_bias, k_cache, v_cache
                )
                k_cache_updates.append(k_cache_update)
                v_cache_updates.append(v_cache_update)
            else:
                x = block(x, attn_bias)[0]

        norm_f = LPLayerNorm[dtype](
            weight("transformer.norm_f.weight"), self.hyperparams
        )
        x = norm_f(x)
        # Generate output tokens using the same SharedEmbedding layer created
        # previously.
        x = wte(x, True)
        if use_cache:
            k_cache = ops.concat(k_cache_updates)
            v_cache = ops.concat(v_cache_updates)
            g.output(List[Symbol](x[-1, axis=1], k_cache, v_cache))
        else:
            g.output(x)
        return g
