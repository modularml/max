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
from math import isqrt, ceil, log2

from max.tensor import Tensor, TensorShape
from max.graph import ops, Dim, Symbol, TensorType, _OpaqueType as OpaqueType
from max.graph.quantization import Float32Encoding, QuantizationEncoding
from max.graph.kv_cache.kernel_names import _kv_cache_kernel_names
from kv_cache.types import (
    ContiguousKVCache,
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

    # scalar layer_idx used to retrieve kv cache objects
    var layer_idx: Symbol

    def __call__(
        self,
        input: Symbol,
        start_pos: Symbol,
        kv_collection: Symbol,
        attn_weight: Symbol,
        valid_lengths: Symbol,
    ) -> Tuple[Symbol, Symbol]:
        """Constructs the forward pass for this attention block.

        input: Activations with shape (batch_size, seq_len, num_heads * head_dim)
        start_pos: Scalar with index of starting token, effectively tracks
            the number of entries in the cache.
        kv_collection: The Collection object containing the KVCache for our layer
        attn_weight: The causal mask and position encoding for this batch
        valid_lengths: The unpadded length of sequences in our batch
        """

        g = input.graph()

        # extract shape characteristics of the input
        full_seq_len = Dim("full_seq_len")
        batch_size, seq_len = input.shape()[0], input.shape()[1]

        # do QKV projections
        xq_type = input.type()
        xq = ops.custom[self._kernel_names.fused_qkv_matmul_kernel](
            List[Symbol](input, self.wqkv, kv_collection, self.layer_idx),
            xq_type,
        )

        seq_len_sym = ops.shape_of(input)[1]
        s_k = start_pos + seq_len_sym

        # reshape Q
        xq = xq.reshape(
            batch_size, seq_len, self.n_heads, int(kv_params.head_size)
        )

        # Calculate out mask
        var output_type = xq.type()
        attn_out = ops.custom[self._kernel_names.flash_attention_kernel](
            List[Symbol](
                xq,
                kv_collection,
                self.layer_idx,
                attn_weight,
                valid_lengths,
                g.scalar(isqrt(Float32(kv_params.head_size))),
            ),
            output_type,
        )

        attn_out = attn_out.reshape(batch_size, seq_len, -1)

        # final projection and return
        return self.wo(attn_out), kv_collection
