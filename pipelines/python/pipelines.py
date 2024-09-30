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

import click
import llama3
from huggingface_hub import hf_hub_download
from max.driver import CPU, CUDA
from max.pipelines import TokenGenerator
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.performance_fake import (
    PerformanceFakingTokenGenerator,
    get_performance_fake,
)
from text_streaming import stream_text_to_console
from transformers import AutoTokenizer
from uvicorn import Server

from utils import TextGenerationMetrics, config_to_flag

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


async def serve_token_generator(
    model: TokenGenerator,
    tokenizer: AutoTokenizer,
    max_batch_size: int = 1,
    profile=False,
):
    """Hosts the Llama3 pipeline using max.serve."""
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    pipeline = TokenGeneratorPipeline[llama3.Llama3Context](
        model, tokenizer, max_batch_size
    )
    pipelines = [pipeline]

    app = fastapi_app(settings, debug_settings, pipelines)
    app.dependency_overrides[token_pipeline] = lambda: pipeline

    config = fastapi_config(app=app)
    server = Server(config)
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
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to run the model on the available GPU.",
)
@click.option(
    "--performance-fake",
    type=click.Choice(["none", "no-op", "speed-of-light"]),
    default="none",
    help="Fake the engine performance (for benchmarking)",
)
def run_llama3(
    prompt, serve, profile_serve, use_gpu, performance_fake, **config_kwargs
):
    """Runs the Llama3 pipeline."""
    if use_gpu:
        config_kwargs.update(
            {
                "device": CUDA(),
                "quantization_encoding": llama3.SupportedEncodings.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device": CPU()})
    config = llama3.InferenceConfig(**config_kwargs)
    # By default, use the Modular HF repository as a reference for tokenizer
    # configuration, etc. when no repository is specified.
    repo_id = f"modularai/llama-{config.version}"
    if config.weight_path is None:
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
        print("Starting server...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        if performance_fake == "none":
            model = llama3.Llama3(config)
        else:
            model = get_performance_fake(tokenizer, performance_fake)

        asyncio.run(
            serve_token_generator(
                model, tokenizer, config.batch_size, profile_serve
            )
        )
    else:
        with TextGenerationMetrics(print_report=True) as metrics:
            model = llama3.Llama3(config)
            print("Beginning text generation...")
            asyncio.run(
                stream_text_to_console(
                    model,
                    prompt,
                    metrics=metrics,
                    max_batch_size=config.batch_size,
                )
            )


if __name__ == "__main__":
    main()
