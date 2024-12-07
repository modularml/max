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
"""Build a Llama3 model via Graph API from Safetensor weights."""

from typing import Optional, Union

from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import SafetensorWeights, Weights
from max.pipelines import PipelineConfig, SupportedEncoding
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
)
from nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    OptimizedRotaryEmbedding,
    RMSNorm,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)


def feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
):
    return MLP(
        linear(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.mlp.gate_proj,
        ),
        linear(
            dtype,
            quantization_encoding,
            hidden_dim,
            feed_forward_length,
            weights.mlp.down_proj,
        ),
        linear(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.mlp.up_proj,
        ),
    )


def linear(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    in_features: int,
    out_features: int,
    weights: Weights,
) -> Linear:
    return Linear(
        weights.weight.allocate(
            dtype, [in_features, out_features], quantization_encoding
        )
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.bfloat16, [dims]), eps)


def embedding(
    pipeline_config: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
):
    return Embedding(
        weights.weight.allocate(
            pipeline_config.dtype,
            [vocab_size, hidden_dim],
            pipeline_config.quantization_encoding.quantization_encoding,
        )
    )


def _attention_opaque(
    kv_params,
    pipeline_config: PipelineConfig,
    rope,
    weights,
    layer_idx,
):
    wq = ops.transpose(
        weights.self_attn.q_proj.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    wk = ops.transpose(
        weights.self_attn.k_proj.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )
    wv = ops.transpose(
        weights.self_attn.v_proj.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )

    wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

    return AttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.self_attn.o_proj,
        ),
        rope=rope,
        layer_idx=layer_idx,
    )


def _transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    kv_params: KVCacheParams,
):
    with graph:
        rope_scaling = None

        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=pipeline_config.huggingface_config.max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=False,
        )

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.model.layers[i],
                    layer_idx=ops.constant(i, DType.uint32),
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.quantization_encoding.quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.model.layers[i],
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.model.layers[i].input_layernorm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.model.layers[i].post_attention_layernorm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.model.embed_tokens,
        )

        output = Linear(
            weights.lm_head.weight.allocate(
                pipeline_config.dtype,
                [
                    pipeline_config.huggingface_config.vocab_size,
                    pipeline_config.huggingface_config.hidden_size,
                ],
                pipeline_config.quantization_encoding.quantization_encoding,
            )
        )

        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        return Transformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.rms_norm_eps,
                weights.model.norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
        )


def attention(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: Union[OptimizedRotaryEmbedding, RotaryEmbedding],
    weights: Weights,
):
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    return NaiveAttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        dim=pipeline_config.huggingface_config.hidden_size,
        wk=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            kv_weight_dim,
            pipeline_config.huggingface_config.hidden_size,
            weights.self_attn.k_proj,
        ),
        wv=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            kv_weight_dim,
            pipeline_config.huggingface_config.hidden_size,
            weights.self_attn.v_proj,
        ),
        wq=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.self_attn.q_proj,
        ),
        wo=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.self_attn.o_proj,
        ),
        rope=rope,
    )


def transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    kv_params: KVCacheParams,
):
    if pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return _transformer_opaque(graph, pipeline_config, weights, kv_params)

    with graph:
        rope_scaling = None

        rope = RotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=pipeline_config.huggingface_config.max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=False,
        )

        layers = [
            NaiveTransformerBlock(
                attention=attention(
                    kv_params, pipeline_config, rope, weights.model.layers[i]
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.quantization_encoding.quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.model.layers[i],
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.model.layers[i].input_layernorm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.model.layers[i].post_attention_layernorm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.model.embed_tokens,
        )

        output = Linear(
            weights.lm_head.weight.allocate(
                pipeline_config.dtype,
                [
                    pipeline_config.huggingface_config.vocab_size,
                    pipeline_config.huggingface_config.hidden_size,
                ],
                pipeline_config.quantization_encoding.quantization_encoding,
            )
        )

        return NaiveTransformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.rms_norm_eps,
                weights.model.norm,
            ),
            output=output,
            theta=pipeline_config.huggingface_config.rope_theta,
            embedding=embedding_layer,
        )


def _build_opaque_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    tokens_type = TensorType(DType.int64, shape=["total_seq_len"])
    # NOTE: input_row_offsets_len should be batch_size + 1.
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"]
    )

    kv_cache_args = kv_manager.input_symbols()[0]

    with Graph(
        "coder",
        input_types=[tokens_type, input_row_offsets_type, *kv_cache_args],
    ) as graph:
        model = transformer(
            graph,
            pipeline_config,
            weights,
            kv_params,
        )
        tokens, input_row_offsets, *kv_cache = graph.inputs
        outputs = model(tokens, kv_cache, input_row_offsets=input_row_offsets)
        graph.output(*outputs)
        return graph


def _build_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    if pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return _build_opaque_graph(
            pipeline_config, weights, kv_params, kv_manager
        )

    tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
    attn_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
    )

    kv_inputs = kv_manager.input_symbols()[0]

    with Graph(
        "coder",
        input_types=[
            tokens_type,
            attn_mask_type,
            *kv_inputs,
        ],
    ) as graph:
        model = transformer(
            graph,
            pipeline_config,
            weights,
            kv_params,
        )
        tokens, attention_mask, k_cache, v_cache, start_pos, _ = graph.inputs
        mask_dtype = (
            pipeline_config.dtype
            if pipeline_config.quantization_encoding
            in [
                SupportedEncoding.float32,
                SupportedEncoding.bfloat16,
            ]
            else DType.float32
        )
        logits = model(
            tokens,
            attention_mask.cast(mask_dtype),
            k_cache,
            v_cache,
            start_pos,
        )[0]

        if pipeline_config.enable_echo:
            graph.output(logits[:, -1], logits)
        else:
            graph.output(logits[:, -1])

        return graph
