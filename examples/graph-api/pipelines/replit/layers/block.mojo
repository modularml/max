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
from pathlib import Path
from pipelines.weights.gguf import GGUFFile
from pipelines.weights.loadable_model import LoadableModel

from ..layers.attention import GroupedQueryAttention
from ..layers.linear import Linear
from ..layers.norm import LPLayerNorm
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


struct MPTBlock[T: LoadableModel, dtype: DType]:
    """A block in the MosaicML Pretrained Transformer model.

    Parameters:
        T: LoadableModel class for loading model weights.
        dtype: The DType of the weights and inputs to this block.
    """

    var attn_norm: LPLayerNorm[dtype]
    var attn: GroupedQueryAttention[dtype]
    var ffn_norm: LPLayerNorm[dtype]
    var ffn: MPTMLP

    def __init__(
        inout self,
        inout params: T,
        layer_index: Int,
        g: Graph,
        hyperparams: HyperParams,
    ):
        """Build a MPT Block from the given params and string prefix.

        Args:
            params: LoadableModel class for loading params weights.
            layer_index: Integer index for this layer.
            g: Graph in which to create the weights.
            hyperparams: Model hyperparameters.
        Returns:
            A new MPTBlock object.
        """

        @parameter
        def weight[weight_type: DType = dtype](name: String) -> Symbol:
            return g.constant(params.get[weight_type](name, layer_index))

        self.attn_norm = LPLayerNorm[dtype](
            # GGUF always stores these as float32.
            weight[DType.float32]("attn_norm"),
            hyperparams,
        )
        self.attn = GroupedQueryAttention[dtype](
            hyperparams,
            Linear(weight("attn_qkv")),
            Linear(weight("attn_output")),
        )
        self.ffn_norm = LPLayerNorm[dtype](
            # GGUF always stores these as float32.
            weight[DType.float32]("ffn_norm"),
            hyperparams,
        )
        self.ffn = MPTMLP(
            Linear(weight("ffn_up")),
            Linear(weight("ffn_down")),
        )

    def __call__(
        self,
        input: Symbol,
        attn_bias: Optional[Symbol] = None,
        k_cache: Optional[Symbol] = None,
        v_cache: Optional[Symbol] = None,
    ) -> (Symbol, Symbol, Symbol):
        g = input.graph()
        with g.layer("MPTBlock"):
            a = self.attn_norm(input)
            b, k_cache_update, v_cache_update = self.attn(
                a, attn_bias, k_cache, v_cache
            )
            # Rebind the attention output to the shape of the input to allow the
            # `add` op to correctly set shapes.
            b = b.rebind(input.tensor_type().dims)
            output = input + b
            m = self.ffn_norm(output)
            n = self.ffn(m)
            return output + n, k_cache_update, v_cache_update
