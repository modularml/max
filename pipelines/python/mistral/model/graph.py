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
"""Build a Mistral model via Graph API from Safetensor weights."""

from max.dtype import DType
from max.graph import Graph, TensorType, TensorValue, ops
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
)
from nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
    Transformer,
    TransformerBlock,
)


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return MLP(
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.w1,
        ),
        linear(
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.feed_forward.w2,
        ),
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.w3,
        ),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


def rms_norm(dims: int, eps: float, weights: SafetensorWeights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.bfloat16, [dims]), eps)


def embedding(
    params: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: SafetensorWeights,
):
    return Embedding(
        weights.weight.allocate(
            params.dtype,
            [vocab_size, hidden_dim],
        )
    )


def _attention_opaque(
    kv_params: KVCacheParams,
    params: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: SafetensorWeights,
    layer_idx: int,
):
    kv_weight_dim = (
        params.huggingface_config.head_dim
        * params.huggingface_config.num_key_value_heads
    )

    wq = ops.transpose(
        weights.attention.wq.weight.allocate(
            params.dtype,
            [
                params.huggingface_config.num_attention_heads
                * params.huggingface_config.head_dim,
                params.huggingface_config.hidden_size,
            ],
        ),
        0,
        1,
    )
    wk = ops.transpose(
        weights.attention.wk.weight.allocate(
            params.dtype,
            [kv_weight_dim, params.huggingface_config.hidden_size],
        ),
        0,
        1,
    )
    wv = ops.transpose(
        weights.attention.wv.weight.allocate(
            params.dtype,
            [kv_weight_dim, params.huggingface_config.hidden_size],
        ),
        0,
        1,
    )
    wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

    return AttentionWithRope(
        n_heads=params.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            params.dtype,
            params.huggingface_config.hidden_size,
            params.huggingface_config.num_attention_heads
            * params.huggingface_config.head_dim,
            weights.attention.wo,
        ),
        rope=rope,
        layer_idx=layer_idx,  # type: ignore
    )


def _transformer(
    graph: Graph,
    params: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
):
    with graph:
        rope = OptimizedRotaryEmbedding(
            dim=params.huggingface_config.num_attention_heads
            * params.huggingface_config.head_dim,
            n_heads=params.huggingface_config.num_attention_heads,
            theta=params.huggingface_config.rope_theta,
            max_seq_len=params.max_length,
            rope_scaling=None,
        )

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    params,
                    rope,
                    weights.layers[i],
                    layer_idx=ops.constant(i, DType.uint32),  # type: ignore
                ),
                mlp=feed_forward(
                    params.dtype,
                    params.huggingface_config.hidden_size,
                    params.huggingface_config.intermediate_size,
                    weights.layers[i],
                ),
                attention_norm=rms_norm(
                    params.huggingface_config.hidden_size,
                    params.huggingface_config.rms_norm_eps,
                    weights.layers[i].attention_norm,
                ),
                mlp_norm=rms_norm(
                    params.huggingface_config.hidden_size,
                    params.huggingface_config.rms_norm_eps,
                    weights.layers[i].ffn_norm,
                ),
            )
            for i in range(params.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            params,
            params.huggingface_config.vocab_size,
            params.huggingface_config.hidden_size,
            weights.tok_embeddings,
        )

        output = linear(
            params.dtype,
            params.huggingface_config.vocab_size,
            params.huggingface_config.hidden_size,
            weights.output,
        )

        kv_collection_cls = FetchContinuousBatchingKVCacheCollection

        return Transformer(
            dim=params.huggingface_config.hidden_size,
            n_heads=params.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                params.huggingface_config.hidden_size,
                params.huggingface_config.rms_norm_eps,
                weights.norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
        )


def _build_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    tokens_type = TensorType(DType.int64, shape=["total_seq_len"])
    input_row_offset_type = TensorType(
        DType.uint32, shape=["input_row_offset_len"]
    )

    kv_cache_args = kv_manager.input_symbols()[0]

    with Graph(
        "mistral",
        input_types=[
            tokens_type,
            input_row_offset_type,
            *kv_cache_args,
        ],
    ) as graph:
        model = _transformer(graph, pipeline_config, weights, kv_params)
        tokens, input_row_offset, *kv_cache = graph.inputs
        logits = model(tokens, kv_cache, input_row_offset=input_row_offset)
        graph.output(logits)
        return graph
