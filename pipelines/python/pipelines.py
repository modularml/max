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

import asyncio
import functools
import os

import click
import llama3
import replit
from huggingface_hub import hf_hub_download
from max.driver import DeviceSpec
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import BatchedTokenGeneratorState
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.performance_fake import (
    PerformanceFakingTokenGeneratorTokenizer,
    get_performance_fake,
)
from text_streaming import stream_text_to_console
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from uvicorn import Server

from utils import DevicesOptionType, TextGenerationMetrics, config_to_flag

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


def pipeline_config(
    kv_cache_strategy,
    batch_size: int,  # Also KV-cache size.
    batch_timeout=0.0,
    max_forward_steps: int = 1,
) -> TokenGeneratorPipelineConfig:
    if kv_cache_strategy == KVCacheStrategy.CONTINUOUS:
        return TokenGeneratorPipelineConfig.continuous_heterogenous(
            tg_batch_size=batch_size,
            ce_batch_size=min(32, batch_size),
            ce_batch_timeout=batch_timeout,
            max_forward_steps=max_forward_steps,
        )
    elif kv_cache_strategy == KVCacheStrategy.NAIVE:
        return TokenGeneratorPipelineConfig.dynamic_homogenous(
            batch_size=batch_size,
            batch_timeout=batch_timeout,
            max_forward_steps=max_forward_steps,
        )
    else:
        raise ValueError(
            f"{kv_cache_strategy} caching strategy is not supported by Serving."
        )


async def serve_token_generator(
    config: llama3.InferenceConfig,
    repo_id: str,
    performance_fake,
    prefer_ce_over_tg: bool = True,
    profile: bool = False,
):
    """Hosts the Llama3 pipeline using max.serve."""
    if performance_fake == "none":
        print(f"Starting server using Llama3.")
        tokenizer = llama3.Llama3Tokenizer(
            config,
        )
        assert tokenizer.delegate
        model_factory = functools.partial(
            llama3.Llama3TokenGenerator,
            config,
            tokenizer.delegate.eos_token_id,
            tokenizer.delegate.vocab_size,
        )
        kv_cache_strategy = config.cache_strategy
    else:
        print(f"Starting server using performance fake '{performance_fake}'.")
        tokenizer = PerformanceFakingTokenGeneratorTokenizer(
            AutoTokenizer.from_pretrained(repo_id)
        )
        model_factory = functools.partial(
            get_performance_fake,
            performance_fake,
        )
        kv_cache_strategy = KVCacheStrategy.CONTINUOUS

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    batch_size = config.max_cache_batch_size
    max_forward_steps = config.max_forward_steps
    batch_config = pipeline_config(
        kv_cache_strategy, batch_size, max_forward_steps=max_forward_steps
    )
    print(
        f"Server configured with {kv_cache_strategy} caching with batch size"
        f" {batch_size}."
    )

    # limit the number of inflight requests to just a few more than the number
    # of active slots on the GPU
    request_limit = batch_size + 128
    settings = Settings(api_types=[APIType.OPENAI], request_limit=request_limit)

    model_name = "llama3"
    app = fastapi_app(
        settings,
        debug_settings,
        {
            model_name: BatchedTokenGeneratorState(
                TokenGeneratorPipeline(
                    batch_config, model_name, tokenizer, False
                ),
                model_factory,
            )
        },
    )

    server = Server(fastapi_config(app=app))
    await server.serve()


class ModelGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Model not supported: {cmd_name}\nSupported models: {supported}"
        )


@click.command(cls=ModelGroup)
def main():
    pass


@main.command(name="llama3")
@config_to_flag(llama3.InferenceConfig)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--serve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to serve an OpenAI HTTP endpoint on port 8000.",
)
@click.option(
    "--profile-serve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to enable pyinstrument profiling on the serving endpoint.",
)
@click.option(
    "--use-gpu",
    is_flag=False,
    type=DevicesOptionType(),
    show_default=True,
    default="",
    flag_value="0",
    help=(
        "Whether to run the model on the available GPU. An ID value can be"
        " provided optionally to indicate the device ID to target."
    ),
)
@click.option(
    "--num-warmups",
    type=int,
    default=1,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
@click.option(
    "--performance-fake",
    type=click.Choice(["none", "no-op", "speed-of-light", "vllm"]),
    default="none",
    help="Fake the engine performance (for benchmarking)",
)
@click.option(
    "--disable-prefer-ce-over-tg",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable preference of context encoding over token generation.",
)
def run_llama3(
    prompt,
    serve,
    profile_serve,
    use_gpu,
    num_warmups,
    performance_fake,
    disable_prefer_ce_over_tg,
    **config_kwargs,
):
    """Runs the Llama3 pipeline."""
    if use_gpu:
        config_kwargs.update(
            {
                "device_spec": DeviceSpec.cuda(id=use_gpu[0]),
                "quantization_encoding": llama3.SupportedEncodings.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_spec": DeviceSpec.cpu()})
    config = llama3.InferenceConfig(**config_kwargs)
    # By default, use the Modular HF repository as a reference for tokenizer
    # configuration, etc. when no repository is specified.
    repo_id = f"modularai/llama-{config.version}"
    if config.weight_path is None and performance_fake == "none":
        if config.huggingface_weights is not None:
            components = config.huggingface_weights.split("/")
            assert len(components) == 3, (
                "invalid Hugging Face weight location:"
                f" {config.huggingface_weights}, "
            )
            repo_id = f"{components[0]}/{components[1]}"
            weight_filename = components[2]

        else:
            weight_filename = config.quantization_encoding.hf_model_name(
                config.version
            )

        config.weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=weight_filename,
        )

    if serve:
        asyncio.run(
            serve_token_generator(
                config,
                repo_id,
                performance_fake,
                profile_serve,
                not disable_prefer_ce_over_tg,
            )
        )
    else:
        # Run timed run & print results
        with TextGenerationMetrics(print_report=True) as metrics:
            tokenizer = llama3.Llama3Tokenizer(
                config,
            )
            model = llama3.Llama3TokenGenerator(
                config,
                tokenizer.delegate.eos_token_id,
                tokenizer.delegate.vocab_size,
            )
            # Run warmup iteration with no metrics & printing disabled
            if num_warmups > 0:
                print("Running warmup...")
                for i in range(num_warmups):
                    asyncio.run(
                        stream_text_to_console(
                            model,
                            tokenizer,
                            prompt,
                            metrics=none,
                            print_tokens=false,
                            max_batch_size=config.max_cache_batch_size,
                        )
                    )

            print("Beginning text generation...")
            asyncio.run(
                stream_text_to_console(
                    model,
                    tokenizer,
                    prompt,
                    metrics=metrics,
                    max_batch_size=config.max_cache_batch_size,
                    n_duplicate=config.n_duplicate,
                )
            )


async def serve_token_generator_replit(
    config: replit.InferenceConfig,
    repo_id: str,
    performance_fake,
    prefer_ce_over_tg: bool = True,
    profile: bool = False,
):
    """Hosts the Replit pipeline using max.serve."""
    if performance_fake == "none":
        print("Starting server using Replit.")
        tokenizer = replit.ReplitTokenizer(
            config,
        )
        assert tokenizer.delegate
        model_factory = functools.partial(
            replit.Replit,
            config,
        )
        kv_cache_strategy = config.cache_strategy
    else:
        print(f"Starting server using performance fake '{performance_fake}'.")
        tokenizer = PerformanceFakingTokenGeneratorTokenizer(
            AutoTokenizer.from_pretrained(repo_id)
        )
        model_factory = functools.partial(
            get_performance_fake,
            performance_fake,
        )
        kv_cache_strategy = KVCacheStrategy.CONTINUOUS

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    batch_size = config.max_cache_batch_size
    max_forward_steps = config.max_forward_steps
    batch_config = pipeline_config(
        kv_cache_strategy, batch_size, max_forward_steps=max_forward_steps
    )
    print(
        f"Server configured with {kv_cache_strategy} caching with batch size"
        f" {batch_size}."
    )

    # limit the number of inflight requests to just a few more than the number
    # of active slots on the GPU
    request_limit = batch_size + 128
    settings = Settings(api_types=[APIType.OPENAI], request_limit=request_limit)

    model_name = "llama3"
    app = fastapi_app(
        settings,
        debug_settings,
        {
            model_name: BatchedTokenGeneratorState(
                TokenGeneratorPipeline(
                    batch_config, model_name, tokenizer, False
                ),
                model_factory,
            )
        },
    )

    server = Server(fastapi_config(app=app))
    await server.serve()


@main.command(name="replit")
@config_to_flag(replit.InferenceConfig)
@click.option(
    "--prompt",
    type=str,
    default='def hello():\n  print("hello world")\n',
    help="The text prompt to use for further generation.",
)
@click.option(
    "--serve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to serve an OpenAI HTTP endpoint on port 8000.",
)
@click.option(
    "--use-gpu",
    is_flag=False,
    type=DevicesOptionType(),
    show_default=True,
    default="",
    flag_value="0",
    help=(
        "Whether to run the model on the available GPU. An ID value can be"
        " provided optionally to indicate the device ID to target."
    ),
)
@click.option(
    "--performance-fake",
    type=click.Choice(["none", "no-op", "speed-of-light", "vllm"]),
    default="none",
    help="Fake the engine performance (for benchmarking)",
)
@click.option(
    "--profile-serve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to enable pyinstrument profiling on the serving endpoint.",
)
@click.option(
    "--server-batch-mode",
    type=click.Choice(["dynamic", "continuous"]),
    default="dynamic",
    help="Configures the servers batching scheme",
)
def run_replit(
    prompt,
    serve,
    use_gpu,
    performance_fake,
    profile_serve,
    server_batch_mode,
    **config_kwargs,
):
    """Runs the Replit pipeline."""
    if use_gpu:
        config_kwargs.update(
            {
                "device_spec": DeviceSpec.cuda(id=use_gpu[0]),
                "quantization_encoding": replit.SupportedEncodings.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_spec": DeviceSpec.cpu()})

    config = replit.InferenceConfig(**config_kwargs)

    if serve:
        print("Starting server...")
        asyncio.run(
            serve_token_generator_replit(
                config,
                "modularai/replit-code-1_5",
                performance_fake,
            )
        )
    else:
        with TextGenerationMetrics(print_report=True) as metrics:
            model = replit.Replit(config)
            tokenizer = replit.ReplitTokenizer(config)
            print("Beginning text generation...")
            asyncio.run(
                stream_text_to_console(
                    model,
                    tokenizer,
                    prompt,
                    metrics=metrics,
                    max_batch_size=config.max_cache_batch_size,
                    n_duplicate=config.n_duplicate,
                )
            )


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
