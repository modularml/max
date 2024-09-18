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
from huggingface_hub import hf_hub_download
from max.driver import CPU, CUDA
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from transformers import AutoTokenizer
from uvicorn import Server

import llama3
from text_streaming import stream_text_to_console
from text_streaming.interfaces import TokenGenerator
from utils import TextGenerationMetrics, config_to_flag

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


async def serve_token_generator(
    model: TokenGenerator, tokenizer: AutoTokenizer, profile=False
):
    """Hosts the Llama3 pipeline using max.serve."""
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings(profiling_enabled=profile)

    pipeline = TokenGeneratorPipeline[llama3.Llama3Context](model, tokenizer)
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
def run_llama3(prompt, serve, profile_serve, use_gpu, **config_kwargs):
    """Runs the Llama3 pipeline."""
    device = CUDA() if use_gpu else CPU()
    config_kwargs.update({"device": device})
    config = llama3.InferenceConfig(**config_kwargs)
    repo_id = f"modularai/llama-{config.version}"
    config.weight_path = hf_hub_download(
        repo_id=repo_id,
        filename=config.quantization_encoding.hf_model_name(config.version),
    )

    if serve:
        print("Starting server...")
        model = llama3.Llama3(config)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        asyncio.run(serve_token_generator(model, tokenizer, profile_serve))
    else:
        with TextGenerationMetrics(print_report=True) as metrics:
            model = llama3.Llama3(config)
            print("Beginning text generation...")
            asyncio.run(
                stream_text_to_console(
                    model,
                    prompt,
                    metrics=metrics,
                )
            )


if __name__ == "__main__":
    main()
