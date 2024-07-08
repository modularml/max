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
"""The core Transformer block of the model."""

from collections import Optional

from max.graph import ops, Symbol, Graph

from ..layers.attention import GroupedQueryAttention
from ..layers.linear import Linear
from ..layers.norm import LPLayerNorm
from ..weights.replit_checkpoint import Checkpoint
from ..weights.hyperparams import HyperParams


@value
struct MPTMLP:
    """Multiplayer perceptron used in MPT."""

    var up_proj: Linear
    var down_proj: Linear

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        with g.layer("MPTMLP"):
            return self.down_proj(ops.gelu(self.up_proj(input)))


struct MPTBlock[dtype: DType]:
    """A block in the MosaicML Pretrained Transformer model.

    Parameters:
        dtype: The DType of the weights and inputs to this block.
    """

    var norm_1: LPLayerNorm[dtype]
    var attn: GroupedQueryAttention[dtype]
    var norm_2: LPLayerNorm[dtype]
    var ffn: MPTMLP

    def __init__(
        inout self,
        norm_1: LPLayerNorm[dtype],
        attn: GroupedQueryAttention[dtype],
        norm_2: LPLayerNorm[dtype],
        ffn: MPTMLP,
    ):
        self.norm_1 = norm_1
        self.attn = attn
        self.norm_2 = norm_2
        self.ffn = ffn

    def __call__(
        self,
        input: Symbol,
        attn_bias: Optional[Symbol] = None,
        k_cache: Optional[Symbol] = None,
        v_cache: Optional[Symbol] = None,
    ) -> (Symbol, Symbol, Symbol):
        g = input.graph()
        with g.layer("MPTBlock"):
            a = self.norm_1(input)
            b, k_cache_update, v_cache_update = self.attn(
                a, attn_bias, k_cache, v_cache
            )
            # Rebind the attention output to the shape of the input to allow the
            # `add` op to correctly set shapes.
            b = b.rebind(input.tensor_type().dims)
            output = input + b
            m = self.norm_2(output)
            n = self.ffn(m)
            return output + n, k_cache_update, v_cache_update

    @staticmethod
    def create[
        T: Checkpoint
    ](
        params: T,
        prefix: String,
        g: Graph,
        hyperparams: HyperParams,
    ) -> MPTBlock[dtype]:
        """Build a MPT Block from the given params and string prefix.

        Args:
            params: Checkpoint class for getting the weights.
            prefix: String prefix to add to the weight key.
            g: Graph in which to create the weights.
            hyperparams: Model hyperparameters.
        Returns:
            A new MPTBlock object.
        """

        @parameter
        def load_param(name: String) -> Symbol:
            return g.constant(params.get[dtype](prefix + name))

        norm_1 = LPLayerNorm[dtype](load_param("norm_1.weight"), hyperparams)
        attn = GroupedQueryAttention[dtype](
            hyperparams,
            Linear(load_param("attn.Wqkv.weight")),
            Linear(load_param("attn.out_proj.weight")),
        )
        norm_2 = LPLayerNorm[dtype](load_param("norm_2.weight"), hyperparams)
        ffn = MPTMLP(
            Linear(load_param("ffn.up_proj.weight")),
            Linear(load_param("ffn.down_proj.weight")),
        )
        return MPTBlock[dtype](norm_1, attn, norm_2, ffn)
