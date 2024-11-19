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
"""Utilities for generating text in the cli."""

import asyncio
import logging
from max.pipelines import PipelineConfig, PIPELINE_REGISTRY
from text_streaming import stream_text_to_console
from ..metrics import TextGenerationMetrics

logger = logging.getLogger(__name__)


def generate_text_for_pipeline(
    pipeline_config: PipelineConfig, prompt: str, num_warmups: int = 0
):
    # Load tokenizer and Pipeline.
    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)

    # Run timed run & print results.
    with TextGenerationMetrics(print_report=True) as metrics:
        # Run warmups if requested.
        if num_warmups > 0:
            logger.info("Running warmup...")
            for _ in range(num_warmups):
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        metrics=None,
                        print_tokens=False,
                        max_batch_size=pipeline_config.max_cache_batch_size,
                    )
                )

        # Run and print results.
        logger.info("Beginning text generation...")
        asyncio.run(
            stream_text_to_console(
                pipeline,
                tokenizer,
                prompt,
                metrics=metrics,
                print_tokens=True,
                max_batch_size=pipeline_config.max_cache_batch_size,
            )
        )
