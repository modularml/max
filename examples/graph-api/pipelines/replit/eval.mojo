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
"""Evaluates Replit model."""
from pathlib import cwd, Path
import sys
from time import perf_counter_ns

from max.engine import InferenceSession, Model, TensorMap
from max.tensor import Tensor, TensorShape

from .config import (
    ReplitConfigRegistry,
    get_replit_base_default_config,
    get_replit_model_url,
)
from model.replit import Replit
from weights.hyperparams import get_default
from run import ReplitPipeline, Config
from pipelines.benchmarks.human_eval import HumanEval
from pipelines.tokenizer import AutoTokenizer
from pipelines.configs.registry import ConfigRegistry, ConfigRegistryDict
from pipelines.configs.parse_args import (
    OptionTypeEnum,
    OptionValue,
    parse_args,
    register_pipeline_configs,
)


def dispatch[dtype: DType](config: Config):
    """Dispatches token generation for a model."""

    # Set up the Replit model prepare it for token generation.
    var max_length: Optional[Int] = None
    if "max-length" in config:
        max_length = config.get("max-length")[Int]
    var max_new_tokens: Optional[Int] = None
    if "max-new-tokens" in config:
        max_new_tokens = config.get("max-new-tokens")[Int]
    replit = ReplitPipeline[dtype](
        config.get("model-path")[Path],
        use_gpu=config.get("experimental-use-gpu")[Bool],
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    @parameter
    def generate(prompt: String) -> Tuple[String, Int]:
        _ = replit.reset(prompt)
        output = str("")
        num_tokens = 0
        while True:
            s = replit.next_token()
            if not s:
                break
            output += s.value()
            num_tokens += 1
        return output, num_tokens

    # Run evaluation on problems from HumanEval.
    eval_benchmark = HumanEval()
    start_time = perf_counter_ns()
    problems = eval_benchmark.get_problems()
    for task_id in problems:
        print("Running task ", task_id, "...", sep="", end="")
        problem_start_time = perf_counter_ns()
        for _ in range(config.get("eval-samples")[Int]):
            completion, num_tokens = generate(problems[task_id]["prompt"])
            eval_benchmark.add_sample(task_id, completion)
            duration = (perf_counter_ns() - problem_start_time) / 1e9
            print(
                " done. Took ",
                duration,
                " seconds to generate ",
                num_tokens,
                " tokens (",
                num_tokens / duration,
                " tokens per second)",
                sep="",
            )

    full_duration = (perf_counter_ns() - start_time) / 1e9
    print("All samples finished, took", full_duration, "seconds to generate.")

    # Export the output samples which can be executed to get the eval score.
    output_path = config.get("output-path")[Path]
    eval_benchmark.write(output_path)
    print(
        "Solution samples written to "
        + output_path.__fspath__()
        + ".\n\nRun `evaluate_functional_correctness "
        + output_path.__fspath__()
        + "` to get the"
        " evaluation scores. If this is your first run, you will need to"
        " uncomment the `exec` line in human_eval/execution.py."
    )

    _ = replit^


def replit_eval():
    additional_args = ConfigRegistryDict()
    additional_args["eval-samples"] = OptionTypeEnum.INT
    additional_args["output-path"] = OptionTypeEnum.STRING
    additional_defaults = Dict[String, OptionValue]()
    additional_defaults["eval-samples"] = 1
    additional_defaults["output-path"] = cwd().joinpath("samples.jsonl")
    config = Config(additional_args, additional_defaults)

    @parameter
    if not is_x86():
        dispatch[DType.float32](config)
    else:
        if config.dtype == DType.bfloat16:
            dispatch[DType.bfloat16](config)
        else:
            dispatch[DType.float32](config)
