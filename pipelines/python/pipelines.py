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
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from text_streaming import stream_text_to_console
from text_streaming.interfaces import TokenGenerator
from uvicorn import Server

from utils import (
    TextGenerationMetrics,
    config_to_flag,
    download_to_cache,
    find_in_cache,
)

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass


async def serve_token_generator(model: TokenGenerator):
    """Hosts the Llama3 pipeline using max.serve."""
    settings = Settings(api_types=[APIType.OPENAI])
    pipeline = TokenGeneratorPipeline[llama3.Llama3Context](model)
    pipelines = [pipeline]
    app = fastapi_app(settings, pipelines)
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
    "--verify",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Whether to verify the SHA of the weights before continuing (warning:"
        " this can add 30-60 seconds of call latency). If SHA checksum fails,"
        " pipeline will warn and then exit"
    ),
)
@click.option(
    "--serve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to serve an OpenAI HTTP endpoint on port 8000.",
)
def run_llama3(prompt, verify, serve, **config_kwargs):
    """Runs the Llama3 pipeline."""
    config = llama3.InferenceConfig(**config_kwargs)
    validate_weight_path(config, verify)

    if serve:
        print("Starting server...")
        model = llama3.Llama3(config)
        asyncio.run(serve_token_generator(model))
    else:
        with TextGenerationMetrics(print_report=True) as metrics:
            model = llama3.Llama3(config)
            print("Beginning text generation...")
            asyncio.run(stream_text_to_console(model, prompt, metrics))


def validate_weight_path(config, verify):
    """Ensures that the config `weight_path` points to a valid local path."""
    if config.serialized_model_path:
        # Although the serialized model already contains weights, the weights
        # file is required for the Llama3 tokenizer. Any valid checkpoint works,
        # so we use the `find_in_cache` method to look for a valid checkpoint
        # before downloading one.
        valid_weight_urls = llama3.PRETRAINED_MODEL_WEIGHTS[config.version]
        config.weight_path = find_in_cache(
            config.weight_path,
            verify=verify,
            default_url=valid_weight_urls[llama3.SupportedEncodings.q4_k],
            valid_urls=valid_weight_urls.values(),
        )
    else:
        config.weight_path = download_to_cache(
            config.remote_weight_location(), verify=verify
        )
    assert config.weight_path is not None


if __name__ == "__main__":
    main()
