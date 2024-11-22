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
import coder
import architectures

__all__ = ["architectures"]

from llama3.model import Llama3Model
import mistral
from coder.config import get_coder_huggingface_files
from huggingface_hub import hf_hub_download
from llama3.config import get_llama_huggingface_file
from llama3.model import Llama3Model
from max.driver import DeviceSpec
from max.pipelines import (
    HuggingFaceFile,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import BatchedTokenGeneratorState
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.performance_fake import (
    PerformanceFakingPipelineTokenizer,
    get_performance_fake,
)
from opentelemetry import trace

# We do not need to use this, but the import is needed to ensure that replit
# gets added to the pipeline registry.
from replit import ReplitModel
from transformers import AutoTokenizer
from uvicorn import Server

from utils.cli import (
    DevicesOptionType,
    TextGenerationMetrics,
    batch_config_from_pipeline_config,
    config_to_flag,
    generate_text_for_pipeline,
    pipeline_config_options,
    serve_pipeline,
    stream_text_to_console,
)

logger = logging.getLogger(__name__)
try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


async def serve_token_generator(
    config: PipelineConfig,
    repo_id: str,
    performance_fake,
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
    batch_config = batch_config_from_pipeline_config(pipeline_config=config)
    logger.info(
        "Server configured with %s caching with batch size %s",
        kv_cache_strategy,
        batch_size,
    )

    settings = Settings(api_types=[APIType.OPENAI])

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    app = fastapi_app(
        settings,
        debug_settings,
        {
            model_name: BatchedTokenGeneratorState(
                TokenGeneratorPipeline(batch_config, model_name, tokenizer),
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
@pipeline_config_options
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
def run_llama3(
    prompt,
    serve,
    profile_serve,
    num_warmups,
    performance_fake,
    **config_kwargs,
):
    """Runs the Llama3 pipeline."""

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

    if len(config_kwargs["weight_path"]) == 0 and performance_fake == "none":
        hf_file = get_llama_huggingface_file(
            config_kwargs["version"], config_kwargs["quantization_encoding"]
        )
        config_kwargs["weight_path"] = [hf_file.download()]

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
def run_llama_vision(
    use_gpu,
    **config_kwargs,
):
    """Runs the Llama3.2 vision pipeline."""
    if use_gpu:
        config_kwargs.update(
            {
                "device_spec": DeviceSpec.cuda(id=use_gpu[0]),
                "quantization_encoding": SupportedEncoding.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_spec": DeviceSpec.cpu()})

    config_kwargs["huggingface_repo_id"] = "meta-llama/Llama-3.2-11B-Vision"
    config_kwargs["quantization_encoding"] = SupportedEncoding.bfloat16
    config = PipelineConfig(**config_kwargs)
    weight_filenames = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]
    config.weight_path = [
        hf_hub_download(repo_id=config.huggingface_repo_id, filename=filename)
        for filename in weight_filenames
    ]

    # TODO: Further implementation here - hook up serving / local CLI workflow.
    # Setting session to None for now.
    model, kv_manager = llama_vision.load_llama_vision_and_kv_manager(
        config=config, session=None
    )


async def serve_token_generator_mistral(
    config: PipelineConfig,
    repo_id: str,
    performance_fake,
    profile: bool = False,
):
    """Hosts the Mistral pipeline using max.serve."""
    if performance_fake == "none":
        print("Starting server using Mistral.")
        tokenizer = TextTokenizer(config)
        assert tokenizer.delegate
        model_factory = functools.partial(
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
    batch_config = batch_config_from_pipeline_config(pipeline_config=config)
    print(
        f"Server configured with {kv_cache_strategy} caching with batch size"
        f" {batch_size}."
    )

    settings = Settings(api_types=[APIType.OPENAI])

    model_name = "mistral"
    app = fastapi_app(
        settings,
        debug_settings,
        {
            model_name: BatchedTokenGeneratorState(
                TokenGeneratorPipeline(batch_config, model_name, tokenizer),
                model_factory,
            )
        },
    )

    server = Server(fastapi_config(app=app))
    await server.serve()


@main.command(name="mistral")
@pipeline_config_options
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
def run_mistral(
    prompt,
    serve,
    performance_fake,
    profile_serve,
    **config_kwargs,
):
    """Runs the Mistral pipeline."""
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

    if len(config.weight_path) == 0:
        hf_file = HuggingFaceFile(
            "mistralai/Mistral-Nemo-Instruct-2407", "consolidated.safetensors"
        )
        config.weight_path = [hf_file.download()]

    if serve:
        logger.info("Starting server...")
        asyncio.run(
            serve_token_generator_mistral(
                config,
                config.huggingface_repo_id,
                performance_fake,
                profile=profile_serve,
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
        return func(*args, **kwargs)

    return wrapper


@main.command(name="serve")
@pipeline_config_options
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


@main.command(name="generate")
@pipeline_config_options
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
def run_pipeline(prompt, num_warmups, **config_kwargs):
    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config, prompt=prompt, num_warmups=num_warmups
    )


@main.command(name="replit")
@pipeline_config_options
@common_server_options
@click.option(
    "--serve",
    type=bool,
    is_flag=True,
    default=False,
    help="Should the pipeline be served.",
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
    # Update basic parameters.
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


@main.command(name="coder")
@pipeline_config_options
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
    "--naive",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to use naive KV caching.",
)
def run_coder(
    prompt,
    serve,
    naive,
    **config_kwargs,
):
    """Runs the Coder pipeline."""
    config_kwargs.update(
        {
            "version": "1.5",
            "huggingface_repo_id": (
                "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            ),
            "quantization_encoding": SupportedEncoding.bfloat16,
        }
    )

    config = PipelineConfig(**config_kwargs)

    if len(config.weight_path) == 0:
        hf_files = get_coder_huggingface_files(
            config.version, config.quantization_encoding
        )
        config.weight_path = [hf_file.download() for hf_file in hf_files]

    if naive or config.quantization_encoding not in [
        SupportedEncoding.bfloat16,
        SupportedEncoding.float32,
    ]:
        config.cache_strategy = KVCacheStrategy.NAIVE

    if serve:
        pass
    else:
        # Run timed run & print results
        with TextGenerationMetrics(print_report=True) as metrics:
            tokenizer = TextTokenizer(
                config,
            )
            model = coder.coder_token_gen.CoderTokenGenerator(
                config,
                tokenizer.delegate.eos_token_id,
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


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
