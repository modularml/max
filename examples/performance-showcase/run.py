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

import argparse
import os

import common
import printouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=["roberta", "clip"],
        help="Choose from one of these models",
        required=True,
    )
    args = parser.parse_args()

    common.clear_results()

    common.test_requirements(args.model)

    print(
        "Doing some one time setup. This takes 5 minutes or so, depending on"
        " the model."
    )
    print("Get a cup of coffee and we'll see you in a minute!")
    print()

    os.makedirs(".cache", exist_ok=True)
    commands = [
        f"python3 generate_input.py -m {args.model}",
        f"python3 run_max.py -m {args.model}",
    ]

    common.shell(
        commands, print_progress=True, env={"TF_CPP_MIN_LOG_LEVEL": "9"}
    )
    print("\nStarting inference throughput comparison")

    printouts.print_sys_info()

    print("\nRunning with PyTorch")
    common.shell([f"python3 run_pt.py -m {args.model}"], stdout=None)

    print("\nRunning with MAX Engine")
    common.shell([f"python3 run_max.py -m {args.model}"], stdout=None)

    # Summary table
    results = common.load_results()
    print("\n====== Speedup Summary ======\n")

    printouts.print_speedup_summary(results, args.model)


if __name__ == "__main__":
    main()
