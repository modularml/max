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
from time import now

from max.engine import InferenceSession, Model, TensorMap
from max.tensor import Tensor, TensorShape

from .model.replit import Replit
from .weights.replit_checkpoint import ReplitCheckpoint
from .weights.hyperparams import get_default
from .run import ReplitPipeline
from ..benchmarks.human_eval import HumanEval
from ..tokenizer import AutoTokenizer
from ..configs.registry import ConfigRegistry, ConfigRegistryDict
from ..configs.parse_args import (
    OptionValue,
    parse_args,
    register_pipeline_configs,
)


@value
struct Config:
    """Configuration for token generation runtime options."""

    var config: Dict[String, OptionValue]

    def __init__(inout self):
        # Add Replit eval specific arguments to config_registry
        replit_additional_configs = ConfigRegistryDict()
        """The number of times to each task from the evaluation dataset."""
        replit_additional_configs["eval-samples"] = OptionTypeEnum.INT
        """Path to write output samples."""
        # TODO: This should probably be made a Path type.
        replit_additional_configs["output-path"] = OptionTypeEnum.STRING

        config_registry = ReplitConfigRegistry(replit_additional_configs)
        default_configs = get_replit_base_default_config()
        default_configs["eval-samples"] = 1
        default_configs["output-path"] = str(cwd().joinpath("samples.jsonl"))

        self.config = register_pipeline_configs(
            config_registry.registry,
            parse_args(),
            default_configs,
        )

        if len(str(self.config["converted-weights-path"])) == 0:
            self.config["converted-weights-path"] = cwd().joinpath(
                ".cache/replit/converted"
            )


def replit_eval():
    config = Config()

    # Set up the Replit model prepare it for token generation.
    var max_length: Optional[Int] = None
    if "max-length" in config:
        max_length = config.get("max-length")[Int]
    var max_new_tokens: Optional[Int] = None
    if "max-new-tokens" in config:
        max_new_tokens = config.get("max-new-tokens")[Int]
    replit = ReplitPipeline[dtype](
        config.get("converted-weights-path")[Path],
        use_gpu=config.get("experimental-use-gpu")[Bool],
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    @parameter
    def generate(prompt: String) -> Tuple[String, Int]:
        replit.reset(prompt)
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
    start_time = now()
    problems = eval_benchmark.get_problems()
    for task_id in problems:
        print("Running task ", task_id, "...", sep="", end="")
        problem_start_time = now()
        for _ in range(config.get("eval-samples")[Int]):
            completion, num_tokens = generate(problems[task_id]["prompt"])
            eval_benchmark.add_sample(task_id, completion)
            duration = (now() - problem_start_time) / 1e9
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

    full_duration = (now() - start_time) / 1e9
    print("All samples finished, took", full_duration, "seconds to generate.")

    # Export the output samples which can be executed to get the eval score.
    output_path = config.get("output-path")[String]
    eval_benchmark.write(output_path)
    print(
        "Solution samples written to "
        + output_path
        + ".\n\nRun `evaluate_functional_correctness "
        + output_path
        + "` to get the"
        " evaluation scores. If this is your first run, you will need to"
        " uncomment the `exec` line in human_eval/execution.py."
    )

    _ = replit^
