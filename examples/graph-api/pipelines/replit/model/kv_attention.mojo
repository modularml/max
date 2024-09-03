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
from math import rsqrt, ceil, log2

from max.tensor import Tensor, TensorShape
from max.graph import ops, Dim, Symbol, TensorType, _OpaqueType as OpaqueType
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from max.serve.kv_cache.kernel_names import _kv_cache_kernel_names
from kv_cache.types import (
    ContiguousKVCache,
    KVCacheLayout,
    KVCacheStaticParams,
)

from pipelines.nn import Linear
from pipelines.nn.attention import rope


@value
struct KVCacheOptimizedAttention[type: DType, kv_params: KVCacheStaticParams]:
    """Attention block that supports the specialized ContiguousKVCache type."""

    alias _kernel_names = _kv_cache_kernel_names[type, kv_params]()

    # hyperparams
    var n_heads: Int
    var dim: Int

    # weights/projections
    var wqkv: Symbol
    var wo: Linear

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        k_cache: Symbol,
        v_cache: Symbol,
        attn_weight: Symbol,
    ) -> Tuple[Symbol, Symbol, Symbol]:
        """Constructs the forward pass for this attention block.

        input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
        start_pos: Scalar with index of starting token, effectively tracks
            the number of entries in the cache.
        k_cache: Previously computed keys. This is a mo.opaque ContiguousKVCache object
            with logical shape (batch, prev_seq_len, n_kv_heads, head_dim).
        v_cache: Previously computed values. This is a mo.opaque ContiguousKVCache object
            with logical shape (batch, prev_seq_len, n_kv_heads, head_dim).
        """

        g = input.graph()

        # extract shape characteristics of the input
        full_seq_len = Dim("full_seq_len")
        batch_size, seq_len = input.shape()[0], input.shape()[1]
        head_dim = g.scalar[type](kv_params.head_size)

        # do QKV projections
        xq_type = input.type()
        xq = ops.custom[self._kernel_names.fused_qkv_matmul_kernel](
            List[Symbol](input, self.wqkv, k_cache, v_cache), xq_type
        )

        seq_len_sym = ops.shape_of(input)[1]
        s_k = start_pos + seq_len_sym

        # reshape Q
        xq = xq.reshape(batch_size, seq_len, self.n_heads, kv_params.head_size)

        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            # Flash Attention shapes differ on CPU and GPU, we need to
            # transpose on cpu. This'll will be fixed by KERN-626
            xq = xq.swapaxes(1, 2)

        # Calculate out mask
        var output_type = xq.type()
        attn_out = ops.custom[self._kernel_names.flash_attention_kernel](
            List[Symbol](
                xq, k_cache, v_cache, attn_weight, ops.rsqrt(head_dim)
            ),
            output_type,
        )

        # transpose hidden state to (batch_size, seq_len, num_heads * head_dim)
        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            # Flash Attention shapes differ on CPU and GPU, we need to
            # transpose on cpu. This'll will be fixed by KERN-626
            attn_out = attn_out.swapaxes(1, 2)
        attn_out = attn_out.reshape(batch_size, seq_len, -1)

        # final projection and return
        return self.wo(attn_out), k_cache, v_cache
