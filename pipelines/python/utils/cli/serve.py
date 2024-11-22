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
"""Utilities for serving cli."""

import asyncio
import functools
import logging
from typing import Union
from max.pipelines import PipelineConfig, PIPELINE_REGISTRY
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.pipelines.deps import BatchedTokenGeneratorState
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.performance_fake import (
    PerformanceFakingPipelineTokenizer,
    get_performance_fake,
)
from transformers import AutoTokenizer
from uvicorn import Server


logger = logging.getLogger(__name__)


def batch_config_from_pipeline_config(
    pipeline_config: PipelineConfig, batch_timeout: float = 0.0
) -> TokenGeneratorPipelineConfig:
    if pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS:
        batch_config = TokenGeneratorPipelineConfig.continuous_heterogenous(
            tg_batch_size=pipeline_config.max_cache_batch_size,
            ce_batch_size=min(
                pipeline_config.max_cache_batch_size,
                pipeline_config.max_ce_batch_size,
            ),
            ce_batch_timeout=batch_timeout,
            max_forward_steps=pipeline_config.max_num_steps,
        )
    elif pipeline_config.cache_strategy == KVCacheStrategy.NAIVE:
        batch_config = TokenGeneratorPipelineConfig.dynamic_homogenous(
            batch_size=pipeline_config.max_cache_batch_size,
            batch_timeout=batch_timeout,
            max_forward_steps=pipeline_config.max_num_steps,
        )
    else:
        raise ValueError(
            f"{pipeline_config.cache_strategy} caching strategy is not"
            " supported by Serving."
        )

    logger.info(
        "Server configured with %s caching with batch size %s",
        pipeline_config.cache_strategy,
        pipeline_config.max_cache_batch_size,
    )

    return batch_config


def serve_pipeline(
    pipeline_config: PipelineConfig,
    performance_fake: str = "none",
    profile: bool = False,
    batch_timeout: float = 0.0,
    model_name: Union[str, None] = None,
):
    # Retrieve tokenizer and pipeline.
    if performance_fake == "none":
        logger.info(
            f"Starting server using {pipeline_config.huggingface_repo_id}"
        )
        # Load tokenizer and pipeline from PIPELINE_REGISTRY.
        # return_factory, returns a pipeline_factory as opposed to an intialized pipeline.
        # this minimizes the amount of data travelling during server worker initialization.
        # it also minimized the probability of a pickling issue with custom config.
        tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve(
            pipeline_config, return_factory=True
        )
    else:
        logger.info(
            f"Starting server using performance fake {performance_fake}."
        )
        tokenizer = PerformanceFakingPipelineTokenizer(  # type: ignore
            AutoTokenizer.from_pretrained(pipeline_config.huggingface_repo_id)
        )
        pipeline_factory = functools.partial(  # type: ignore
            get_performance_fake,
            performance_fake,  # type: ignore
        )
        pipeline_config.cache_strategy = KVCacheStrategy.CONTINUOUS

    # Initialize settings, and TokenGeneratorPipelineConfig.
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    # Load batch config.
    batch_config = batch_config_from_pipeline_config(
        pipeline_config=pipeline_config,
        batch_timeout=batch_timeout,
    )

    # If explicit model name is not provided, set to huggingface_repo_id.
    if model_name is None:
        model_name = pipeline_config.huggingface_repo_id

    # Intialize and serve webserver.
    app = fastapi_app(
        settings,
        debug_settings,
        {
            model_name: BatchedTokenGeneratorState(  # type: ignore
                TokenGeneratorPipeline(
                    batch_config,
                    pipeline_config.huggingface_repo_id,  # type: ignore
                    tokenizer,
                ),
                pipeline_factory,  # type: ignore
            )
        },
    )

    server = Server(fastapi_config(app=app))
    asyncio.run(server.serve())
