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
from collections import Optional, List
from pathlib import Path

from kv_cache.types import (
    ContiguousKVCacheCollection,
    KVCacheStaticParams,
)
from max.graph import _OpaqueType as OpaqueType
from max.graph import ops, Dim, TensorType, Symbol, Graph, Type
from max.tensor import TensorSpec
from max.driver import AnyTensor, Device
from pipelines.nn import Embedding, Linear
from pipelines.weights.gguf import GGUFFile
from pipelines.weights.loadable_model import LoadableModel
from ..model.block import MPTMLP
from ..model.norm import LPLayerNorm
from ..model.kv_attention import KVCacheOptimizedAttention
from ..model.kv_transformer import (
    KVCacheOptimizedTransformer,
    KVCacheOptimizedTransformerBlock,
)
from ..weights.hyperparams import HyperParams


struct Replit[T: LoadableModel, dtype: DType, kv_params: KVCacheStaticParams]:
    """Replit model implementation.

    Parameters:
        T: LoadableModel class for loading model weights.
        dtype: The DType of the weights and inputs to this model.
        kv_params: KV Cache parameters.
    """

    var hyperparams: HyperParams

    def __init__(out self, hyperparams: HyperParams):
        self.hyperparams = hyperparams

    def build_graph(
        self,
        mut params: T,
        name: String,
    ) -> Graph:
        """Builds the replit model graph.

        The graph takes encoded tokens as input and outputs the predicted
        logits.

        Args:
            params: LoadableModel class for loading model weights.
            name: Name of the graph.
        Returns:
            Replit Graph.
        """
        # Set up graph and inputs.
        seq_len = "seq_len"
        input_type = TensorType(
            DType.int64, self.hyperparams.batch_size, seq_len
        )
        in_types = List[Type](input_type)
        attention_mask_type = TensorType(
            DType.bool, self.hyperparams.batch_size, "full_seq_len"
        )
        in_types.append(attention_mask_type)

        cache_type = OpaqueType(
            ContiguousKVCacheCollection[dtype, kv_params].id()
        )
        in_types.append(cache_type)
        g = Graph(name, in_types=in_types)

        @parameter
        def weight[
            weight_type: DType = dtype
        ](name: String, layer: Optional[Int] = None) -> Symbol:
            return g.constant(params.get[weight_type](name, layer))

        @parameter
        def norm(
            name: String, layer: Optional[Int] = None
        ) -> LPLayerNorm[dtype]:
            return LPLayerNorm[dtype](
                # GGUF always stores these as float32.
                weight[DType.float32](name, layer),
                self.hyperparams,
            )

        layers = List[KVCacheOptimizedTransformerBlock[dtype, kv_params]]()
        for layer in range(self.hyperparams.num_blocks):
            attention = KVCacheOptimizedAttention[dtype, kv_params](
                n_heads=self.hyperparams.n_heads,
                dim=self.hyperparams.d_model,
                wqkv=weight("attn_qkv", layer),
                wo=Linear(weight("attn_output", layer).swapaxes(0, 1)),
                layer_idx=g.scalar(UInt32(layer)),
            )

            feed_forward = MPTMLP(
                Linear(weight("ffn_up", layer).swapaxes(0, 1)),
                Linear(weight("ffn_down", layer).swapaxes(0, 1)),
            )

            layers.append(
                KVCacheOptimizedTransformerBlock[dtype, kv_params](
                    attention=attention,
                    feed_forward=feed_forward,
                    attention_norm=norm("attn_norm", layer),
                    ffn_norm=norm("ffn_norm", layer),
                )
            )

        embedding = Embedding(weight("token_embd"))
        output = embedding.weights
        model = KVCacheOptimizedTransformer[dtype, kv_params](
            dim=self.hyperparams.d_model,
            n_heads=self.hyperparams.n_heads,
            max_seq_len=self.hyperparams.seq_len,
            alibi_bias_max=self.hyperparams.alibi_bias_max,
            embedding=embedding,
            layers=layers,
            norm=norm("output_norm"),
            theta=10000.0,
            output=output.swapaxes(0, 1),
        )

        outputs = model(tokens=g[0], attention_mask=g[1], kv_collection=g[2])
        logits = outputs[0]

        g.output(List[Symbol](logits[-1, axis=1], outputs[1]))

        return g
