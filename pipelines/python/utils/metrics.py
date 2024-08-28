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

"""Metric-gathering utilities for the pipelines."""

import time
from typing import Union


class TextGenerationMetrics:
    """Metrics capturing and reporting for a text generation pipeline."""

    prompt_size: int
    output_size: int
    startup_time: Union[float, str]
    time_to_first_token: Union[float, str]
    prompt_eval_throughput: Union[float, str]
    eval_throughput: Union[float, str]

    _start_time: float
    _signposts: dict[str, float]
    _should_print_report: bool

    def __init__(self, print_report: bool = False):
        self.signposts = {}
        self.prompt_size = 0
        self.output_size = 0
        self._should_print_report = print_report
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._calculate_results()
        if self._should_print_report:
            self._print_report()

    def signpost(self, name: str):
        """Measure the current time and tag it with a name for later reporting.
        """
        self.signposts[name] = time.time()

    def new_token(self):
        """Report that a new token has been generated."""
        self.output_size += 1

    def _calculate_results(self):
        begin_generation = self.signposts.get("begin_generation")
        if begin_generation:
            self.startup_time = (
                self.signposts["begin_generation"] - self.start_time
            ) * 1000.0
        else:
            self.startup_time = "n/a"

        first_token = self.signposts.get("first_token")
        if first_token and begin_generation:
            self.time_to_first_token = (
                self.signposts["first_token"]
                - self.signposts["begin_generation"]
            ) * 1000.0
        else:
            self.time_to_first_token = "n/a"

        end_generation = self.signposts.get("end_generation")
        if end_generation and first_token and begin_generation:
            generation_time = (
                self.signposts["end_generation"] - self.signposts["first_token"]
            )
            assert isinstance(self.time_to_first_token, float)
            self.prompt_eval_throughput = self.prompt_size / (
                self.time_to_first_token / 1000.0
            )
            self.eval_throughput = (self.output_size - 1) / generation_time
        else:
            self.prompt_eval_throughput = "n/a"
            self.eval_throughput = "n/a"

    def _print_report(self):
        print()
        print("Prompt size:", self.prompt_size)
        print("Output size:", self.output_size)
        print("Startup time:", self.startup_time, "ms")
        print("Time to first token:", self.time_to_first_token, "ms")
        print(
            "Prompt eval throughput (context-encoding):",
            self.prompt_eval_throughput,
            "tokens per second",
        )
        print(
            "Eval throughput (token-generation):",
            self.eval_throughput,
            "tokens per second",
        )
