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
from tensor import Tensor, TensorShape

from ..benchmarks.human_eval import HumanEval
from .model.replit import Replit
from .weights.replit_checkpoint import ReplitCheckpoint
from .weights.hyperparams import get_default
from ..tokenizer import AutoTokenizer
from .run import ReplitPipeline

alias DEFAULT_MAX_SEQ_LEN = 512
alias DEFAULT_EVAL_SAMPLES = 1


@value
struct Config:
    """Configuration for token generation runtime options."""

    var converted_weights_path: Path
    var max_length: Optional[Int]
    var max_new_tokens: Optional[Int]
    var use_gpu: Bool

    var eval_samples: Optional[Int]
    """The number of times to each task from the evaluation dataset."""

    var output_path: String
    """Path to write output samples."""

    def __init__(
        inout self,
        /,
        converted_weights_path: Path = "",
        max_length: Optional[Int] = None,
        max_new_tokens: Optional[Int] = None,
        use_gpu: Bool = False,
        eval_samples: Optional[Int] = None,
        output_path: String = "",
    ):
        self.converted_weights_path = converted_weights_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.use_gpu = use_gpu
        self.eval_samples = eval_samples
        self.output_path = output_path
        self.parse_args()

    def parse_args(inout self):
        args = sys.argv()

        @parameter
        def read_value(index: Int) -> StringRef:
            if index >= len(args):
                raise "missing value for parameter `" + str(
                    args[index - 1]
                ) + "`"
            return args[index]

        # Skip the run_pipeline.mojo and replit arguments.
        i = 2

        while i < len(args):
            if args[i] == "--converted_weights_path":
                self.converted_weights_path = Path(read_value(i + 1))
                i += 2
            elif args[i] == "--max_length":
                self.max_length = int(read_value(i + 1))
                i += 2
            elif args[i] == "--max_new_tokens":
                self.max_new_tokens = int(read_value(i + 1))
                i += 2
            elif args[i] == "--eval_samples":
                self.eval_samples = int(read_value(i + 1))
                i += 2
            elif args[i] == "--experimental-use-gpu":
                self.use_gpu = True
                i += 1
            else:
                raise "unsupported CLI argument: " + String(args[i])

        if len(str(self.converted_weights_path)) == 0:
            self.converted_weights_path = cwd().joinpath(
                ".cache/replit/converted"
            )
        if len(self.output_path) == 0:
            self.output_path = str(cwd().joinpath("samples.jsonl"))

    def get_eval_samples(self) -> Int:
        if self.eval_samples:
            return self.eval_samples.value()
        else:
            return DEFAULT_EVAL_SAMPLES


def replit_eval():
    config = Config()

    # Set up the Replit model prepare it for token generation.
    var replit = ReplitPipeline(
        config.converted_weights_path,
        use_gpu=config.use_gpu,
        max_length=config.max_length,
        max_new_tokens=config.max_new_tokens,
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
        for _ in range(config.get_eval_samples()):
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
    output_path = config.output_path
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
