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
from architectures import register_all_models
from cli import (
    TextGenerationMetrics,
    generate_text_for_pipeline,
    pipeline_config_options,
    serve_pipeline,
    stream_text_to_console,
)
from coder.config import get_coder_huggingface_files
from max.pipelines import PipelineConfig, SupportedEncoding, TextTokenizer
from max.pipelines.kv_cache import KVCacheStrategy

logger = logging.getLogger(__name__)
try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


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
    register_all_models()

    pass


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
def cli_pipeline(prompt, num_warmups, **config_kwargs):
    # Load tokenizer & pipeline.
    pipeline_config = PipelineConfig(**config_kwargs)
    generate_text_for_pipeline(
        pipeline_config, prompt=prompt, num_warmups=num_warmups
    )


# All the models.


@main.command(name="llama3")
@pipeline_config_options
@common_server_options
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--num-warmups",
    type=int,
    default=1,
    show_default=True,
    help="# of warmup iterations to run before the final timed run.",
)
@click.option(
    "--serve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to serve an OpenAI HTTP endpoint on port 8000.",
)
def run_llama3(
    prompt,
    num_warmups,
    serve,
    profile_serve,
    performance_fake,
    batch_timeout,
    model_name,
    **config_kwargs,
):
    """Runs the Llama3 pipeline."""

    # Update basic parameters.
    if config_kwargs["architecture"] is None:
        config_kwargs["architecture"] = "LlamaForCausalLM"

    config = PipelineConfig(**config_kwargs)

    if config.quantization_encoding not in [
        SupportedEncoding.bfloat16,
        SupportedEncoding.float32,
    ]:
        config.cache_strategy = KVCacheStrategy.NAIVE

    if serve:
        serve_pipeline(
            pipeline_config=config,
            profile=profile_serve,
            performance_fake=performance_fake,
            batch_timeout=batch_timeout,
            model_name=model_name,
        )
    else:
        generate_text_for_pipeline(
            pipeline_config=config, prompt=prompt, num_warmups=num_warmups
        )


@main.command(name="replit")
@pipeline_config_options
@common_server_options
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
@click.option(
    "--serve",
    type=bool,
    is_flag=True,
    default=False,
    help="Should the pipeline be served.",
)
def replit(
    prompt,
    num_warmups,
    serve,
    profile_serve,
    performance_fake,
    batch_timeout,
    model_name,
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
            config.version,  # type: ignore
            config.quantization_encoding,  # type: ignore
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
