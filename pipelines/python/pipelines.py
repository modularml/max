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
import logging
import os

import click
import llama3
from llama3.model import Llama3Model
import llama3.vision as llama3_vision
import mistral
from typing import Union
from huggingface_hub import hf_hub_download
from llama3.config import get_llama_huggingface_file
from max.driver import DeviceSpec
from max.pipelines import (
    HuggingFaceFile,
    PipelineConfig,
    SupportedEncoding,
    TextTokenizer,
    TextGenerationPipeline,
    PIPELINE_REGISTRY,
)
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
    PerformanceFakingPipelineTokenizer,
    get_performance_fake,
)
from replit.config import get_replit_huggingface_file
from replit.model import ReplitModel
from transformers import AutoTokenizer
from uvicorn import Server
from opentelemetry import trace

from utils import DevicesOptionType, TextGenerationMetrics, config_to_flag
from utils.cli import (
    serve_pipeline,
    generate_text_for_pipeline,
    stream_text_to_console,
)

logger = logging.getLogger(__name__)
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
            ce_batch_size=batch_size,
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
    config: PipelineConfig,
    repo_id: str,
    performance_fake,
    prefer_ce_over_tg: bool = True,
    profile: bool = False,
):
    """Hosts the Llama3 pipeline using max.serve."""
    if performance_fake == "none":
        logger.info("Starting server using Llama3.")
        tokenizer = TextTokenizer(config)
        assert tokenizer.delegate
        model_factory = functools.partial(
            TextGenerationPipeline,
            pipeline_config=config,
            pipeline_model=Llama3Model,
            eos_token_id=tokenizer.eos,
        )
        kv_cache_strategy = config.cache_strategy
    else:
        logger.info(
            "Starting server using performance fake '%s'.", performance_fake
        )
        tokenizer = PerformanceFakingPipelineTokenizer(  # type: ignore
            AutoTokenizer.from_pretrained(repo_id)
        )
        model_factory = functools.partial(  # type: ignore
            get_performance_fake,  # type: ignore
            performance_fake,
        )
        kv_cache_strategy = KVCacheStrategy.CONTINUOUS

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    batch_size = config.max_cache_batch_size
    batch_config = pipeline_config(
        kv_cache_strategy, batch_size, max_forward_steps=config.max_num_steps
    )
    logger.info(
        "Server configured with %s caching with batch size %s",
        kv_cache_strategy,
        batch_size,
    )

    # limit the number of inflight requests to just a few more than the number
    # of active slots on the GPU
    request_limit = batch_size + 128
    settings = Settings(api_types=[APIType.OPENAI], request_limit=request_limit)

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    app = fastapi_app(
        settings,
        debug_settings,
        {
            model_name: BatchedTokenGeneratorState(
                TokenGeneratorPipeline(
                    batch_config, model_name, tokenizer, True
                ),
                model_factory,
            )
        },
    )
    # Export traces to DataDog
    if os.environ.get("MODULAR_ENABLE_TRACING"):
        try:
            from ddtrace.opentelemetry import TracerProvider

            logger.info("Exporting traces to datadog")
            trace.set_tracer_provider(TracerProvider())
        except ImportError:
            logger.info("ddtrace not found. Not exporting traces")
            pass

    server = Server(fastapi_config(app=app))
    await server.serve()


class ModelGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\nSupported commands:"
            f" {supported}"
        )


@click.command(cls=ModelGroup)
def main():
    pass


@main.command(name="llama3")
@config_to_flag(PipelineConfig)
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
                "quantization_encoding": SupportedEncoding.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_spec": DeviceSpec.cpu()})

    if config_kwargs["architecture"] is None:
        config_kwargs["architecture"] = "LlamaForCausalLM"

    # By default, use the Modular HF repository as a reference for tokenizer
    # configuration, etc. when no repository is specified.
    if config_kwargs["version"] is None:
        config_kwargs["version"] = "3.1"

    if config_kwargs["quantization_encoding"] is None:
        config_kwargs["quantization_encoding"] = SupportedEncoding.q4_k

    if config_kwargs["huggingface_repo_id"] is None:
        if config_kwargs["version"] == "3.1":
            config_kwargs["huggingface_repo_id"] = "modularai/llama-3.1"
        elif config_kwargs["version"] == "3":
            config_kwargs["huggingface_repo_id"] = "modularai/llama-3"
        else:
            raise ValueError(
                f"Model version: {config_kwargs['version']} not supported."
            )

    if config_kwargs["weight_path"] is None and performance_fake == "none":
        hf_file = get_llama_huggingface_file(
            config_kwargs["version"], config_kwargs["quantization_encoding"]
        )
        config_kwargs["weight_path"] = hf_file.download()
    elif config_kwargs["weight_path"] is not None:
        if not os.path.exists(config_kwargs["weight_path"]):
            hf_file = HuggingFaceFile.parse(config_kwargs["weight_path"])
            config_kwargs["weight_path"] = hf_file.download()

    config = PipelineConfig(**config_kwargs)

    if config.quantization_encoding not in [
        SupportedEncoding.bfloat16,
        SupportedEncoding.float32,
    ]:
        config.cache_strategy = KVCacheStrategy.NAIVE

    if serve:
        asyncio.run(
            serve_token_generator(
                config,
                config.huggingface_repo_id,
                performance_fake,
                profile_serve,
                not disable_prefer_ce_over_tg,
            )
        )
    else:
        # Run timed run & print results
        with TextGenerationMetrics(print_report=True) as metrics:
            tokenizer = TextTokenizer(
                config,
            )
            model = llama3.Llama3TokenGenerator(
                config,
                tokenizer.delegate.eos_token_id,
            )
            # Run warmup iteration with no metrics & printing disabled
            if num_warmups > 0:
                logger.info("Running warmup...")
                for _ in range(num_warmups):
                    asyncio.run(
                        stream_text_to_console(
                            model,
                            tokenizer,
                            prompt,
                            metrics=None,
                            print_tokens=False,
                        )
                    )

            logger.info("Beginning text generation...")
            asyncio.run(
                stream_text_to_console(
                    model,
                    tokenizer,
                    prompt,
                    metrics=metrics,
                )
            )


# TODO: We run this llama 3 vision model variant as a separate command for now.
# I think there is room to consolidate it under the "llama3" above.
@main.command(name="llama3-vision")
@config_to_flag(llama3_vision.config.InferenceConfig)
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
def run_llama3_vision(
    use_gpu,
    **config_kwargs,
):
    """Runs the Llama3.2 vision pipeline."""
    if use_gpu:
        config_kwargs.update(
            {
                "device_spec": DeviceSpec.cuda(id=use_gpu[0]),
                "quantization_encoding": llama3.SupportedEncodings.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_spec": DeviceSpec.cpu()})

    config = llama3_vision.config.InferenceConfig(**config_kwargs)
    weight_filenames = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]
    config.weight_path = [
        hf_hub_download(repo_id=config.repo_id, filename=filename)
        for filename in weight_filenames
    ]

    # TODO: Further implementation here - hook up serving / local CLI workflow.
    # Setting session to None for now.
    model, kv_manager = llama3_vision.load_llama3_vision_and_kv_manager(
        config=config, session=None
    )


async def serve_token_generator_mistral(
    config: PipelineConfig,
    repo_id: str,
    performance_fake,
    prefer_ce_over_tg: bool = True,
    profile: bool = False,
):
    """Hosts the Mistral pipeline using max.serve."""
    if performance_fake == "none":
        print("Starting server using Mistral.")
        tokenizer = TextTokenizer(config)  # type: ignore
        assert tokenizer.delegate
        model_factory = functools.partial(  # type: ignore
            mistral.Mistral,
            config,
        )
        kv_cache_strategy = config.cache_strategy
    else:
        print(f"Starting server using performance fake '{performance_fake}'.")
        tokenizer = PerformanceFakingPipelineTokenizer(  # type: ignore
            AutoTokenizer.from_pretrained(repo_id)
        )
        model_factory = functools.partial(  # type: ignore
            get_performance_fake,  # type: ignore
            performance_fake,
        )
        kv_cache_strategy = KVCacheStrategy.CONTINUOUS

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    batch_size = config.max_cache_batch_size
    batch_config = pipeline_config(
        kv_cache_strategy, batch_size, max_forward_steps=config.max_num_steps
    )
    print(
        f"Server configured with {kv_cache_strategy} caching with batch size"
        f" {batch_size}."
    )

    # limit the number of inflight requests to just a few more than the number
    # of active slots on the GPU
    request_limit = batch_size + 128
    settings = Settings(api_types=[APIType.OPENAI], request_limit=request_limit)

    model_name = "mistral"
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


@main.command(name="mistral")
@config_to_flag(PipelineConfig)
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
def run_mistral(
    prompt,
    serve,
    use_gpu,
    performance_fake,
    profile_serve,
    server_batch_mode,
    **config_kwargs,
):
    """Runs the Mistral pipeline."""
    if use_gpu:
        config_kwargs.update(
            {
                "device_spec": DeviceSpec.cuda(id=use_gpu[0]),
                "quantization_encoding": SupportedEncoding.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_spec": DeviceSpec.cpu()})

    if config_kwargs["huggingface_repo_id"] is None:
        config_kwargs[
            "huggingface_repo_id"
        ] = "mistralai/Mistral-Nemo-Instruct-2407"

    if config_kwargs["architecture"] is None:
        config_kwargs["architecture"] = "MistralForCausalLM"

    config = PipelineConfig(**config_kwargs)

    # Validate encoding.
    if config.quantization_encoding is None:
        config.quantization_encoding = SupportedEncoding.bfloat16

    if config.quantization_encoding not in [
        SupportedEncoding.bfloat16,
        SupportedEncoding.float32,
    ]:
        config.cache_strategy = KVCacheStrategy.NAIVE

    if config.weight_path is None:
        hf_file = HuggingFaceFile(
            "mistralai/Mistral-Nemo-Instruct-2407", "consolidated.safetensors"
        )
        config.weight_path = hf_file.download()

    if serve:
        logger.info("Starting server...")
        asyncio.run(
            serve_token_generator_mistral(
                config,
                config.huggingface_repo_id,
                performance_fake,
            )
        )
    else:
        with TextGenerationMetrics(print_report=True) as metrics:
            model = mistral.Mistral(config)
            tokenizer = TextTokenizer(config)
            logger.info("Beginning text generation...")
            asyncio.run(
                stream_text_to_console(
                    model,
                    tokenizer,
                    prompt,
                    metrics=metrics,
                )
            )


def common_server_options(func):
    @config_to_flag(PipelineConfig)
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
        "--profile-serve",
        is_flag=True,
        show_default=True,
        default=False,
        help=(
            "Whether to enable pyinstrument profiling on the serving endpoint."
        ),
    )
    @click.option(
        "--performance-fake",
        type=click.Choice(["none", "no-op", "speed-of-light", "vllm"]),
        default="none",
        help="Fake the engine performance (for benchmarking)",
    )
    @click.option(
        "--batch-timeout",
        type=float,
        default=0.0,
        help="Custom timeout for any particular batch.",
    )
    @click.option(
        "--model-name",
        type=str,
        help="Optional explicit name for serving the model.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["use_gpu"]:
            kwargs["device_spec"] = DeviceSpec.cuda(id=kwargs["use_gpu"][0])
            # If the user is passing in a specific, quantization_encoding don't overwrite it.
            # If it is empty, set it to default to bfloat16 on gpu.
            if kwargs["quantization_encoding"] is None:
                kwargs["quantization_encoding"] = SupportedEncoding.bfloat16
        else:
            kwargs["device_spec"] = DeviceSpec.cpu()

        del kwargs["use_gpu"]

        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve")
@common_server_options
def cli_serve(
    profile_serve,
    performance_fake,
    batch_timeout,
    model_name,
    **config_kwargs,
):
    # Initialize config, and serve.
    pipeline_config = PipelineConfig(**config_kwargs)
    serve_pipeline(
        pipeline_config=pipeline_config,
        profile=profile_serve,
        performance_fake=performance_fake,
        batch_timeout=batch_timeout,
        model_name=model_name,
    )


@main.command(name="replit")
@common_server_options
@click.option(
    "--serve",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to generate or serve model.",
)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
def replit(
    profile_serve,
    performance_fake,
    batch_timeout,
    model_name,
    serve,
    prompt,
    num_warmups,
    **config_kwargs,
):
    if config_kwargs["architecture"] is None:
        config_kwargs["architecture"] = "MPTForCausalLM"

    if config_kwargs["architecture"] != "MPTForCausalLM":
        msg = (
            f"provided architecture '{config_kwargs['architecture']}' not"
            " compatible with Replit."
        )
        raise ValueError(msg)

    config_kwargs["trust_remote_code"] = True

    # Initialize config, and serve.
    pipeline_config = PipelineConfig(**config_kwargs)

    if serve:
        serve_pipeline(
            pipeline_config=pipeline_config,
            profile=profile_serve,
            performance_fake=performance_fake,
            batch_timeout=batch_timeout,
            model_name=model_name,
        )
    else:
        generate_text_for_pipeline(
            pipeline_config, prompt=prompt, num_warmups=num_warmups
        )


@main.command(name="generate")
@config_to_flag(PipelineConfig)
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
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
def run_pipeline(use_gpu, prompt, num_warmups, **config_kwargs):
    # Update config_kwargs for use_gpu.
    if use_gpu:
        config_kwargs["device_spec"] = DeviceSpec.cuda(id=use_gpu[0])
        config_kwargs["quantization_encoding"] = SupportedEncoding.bfloat16
    else:
        config_kwargs["device_spec"] = DeviceSpec.cpu()

    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config, prompt=prompt, num_warmups=num_warmups
    )


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
