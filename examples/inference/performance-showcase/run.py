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

import os
import subprocess
import common
import random
from common import shell
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["roberta", "clip"], help="Choose from one of these models", required=True)
args = parser.parse_args()

common.clear_results()

common.test_requirements(args.model)

print(
    "Doing some one time setup. This takes 5 minutes or so, depending on the model."
)
print("Get a cup of coffee and we'll see you in a minute!")
print()

os.makedirs(".cache", exist_ok=True)
commands = [
    f"python3 generate_input.py {args.model}",
    f"python3 run_tf.py {args.model}",
    f"python3 run_max.py {args.model}",
]

shell(commands, print_progress=True, env={"TF_CPP_MIN_LOG_LEVEL": "9"})
print("\nStarting inference throughput comparison")
info_message = "-" * 40 + "System Info" + "-" * 40
print(f"\n{info_message}\n")
subprocess.run(["cpuinfo", "-g"])
print("\n" + "-" * len(info_message))

print("\nRunning with TensorFlow")
shell([f"python3 run_tf.py {args.model}"], stdout=None)

print("\nRunning with PyTorch")
shell([f"python3 run_pt.py {args.model}"], stdout=None)

print("\nRunning on the Max Engine")
shell([f"python3 run_max.py {args.model}"], stdout=None)

# Summary table
results = common.load_results()
print("\n====== Speedup Summary ======")
from itertools import cycle

exclamations = cycle(["ZAP!", "SHAZAM!", "KAPOW!", "BANG!", "WHAM!"])
[next(exclamations) for i in range(random.randint(0, 4))]

print(
    f'Modular ({results["max"]:.2f} QPS) > TensorFlow ({results["tf"]:.2f} QPS). {next(exclamations)} It\'s {results["max"]/results["tf"]:.2f}x faster!'
)

if "pt" in results:
    print(
        f'Modular ({results["max"]:.2f} QPS) > PyTorch ({results["pt"]:.2f} QPS). {next(exclamations)} It\'s {results["max"]/results["pt"]:.2f}x faster!'
    )
    print()
else:
    pass
