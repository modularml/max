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

from pathlib import Path

import sys
import warnings
warnings.simplefilter("ignore")

import signal
import random
from itertools import cycle
import subprocess
import time
import os

# suppress extraneous logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "9"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    import transformers
    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()
except ModuleNotFoundError:
    pass

NUM_ITERS = 90

def test_requirements(selected_model):
    if selected_model == "clip":
        try:
            import torch
        except ModuleNotFoundError:
            print("\nError: PyTorch not found but is required for clip. Please python3 -m pip install torch")
            print()
            exit(1)

    try:
        import PIL 
    except ModuleNotFoundError:
        print("\nError: Pillow not found but is required. Please python3 -m pip install -r requirements.txt")
        print()
        exit(1)

    try:
        import tensorflow
    except ModuleNotFoundError:
        print("\nError: TensorFlow not found but is required. Please python3 -m pip install -r requirements.txt")
        print()
        exit(1)

    try:
        import transformers
    except ModuleNotFoundError:
        print("\nError: HuggingFace Transformers library not found but is required. Please python3 -m pip install -r requirements.txt")
        print()
        exit(1)

    try:
        from max import engine
    except ModuleNotFoundError:
        print("\nError: Max Engine not found but is required. Please follow the README instructions in the repository root to install the Max Engine python wheel")
        print()
        exit(1)



def clear_results():
    if os.path.exists(".cache/results"):
        os.remove(".cache/results")


def save_result(framework, qps):
    with open(".cache/results", "a") as f:
        f.write(f"{framework},{qps}\n")


def load_results():
    with open(".cache/results", "r") as f:
        results = {}
        for line in f.readlines():
            split = line.split(",")
            results[split[0]] = float(split[1])
    return results


def shell(commands, stdout=subprocess.PIPE, print_progress=False, env={}):
    wait_strings = [
        "Initializing Mojo compiler...",
        "Loading AI empathy modules...",
        "Training AI on basic human interactions...",
        "Reticulating splines...",
        "Debugging Mojo humor algorithms...",
        "You are being admirably patient...",
        "Optimizing neural networks for efficiency...",
        "Executing advanced Mojo functions...",
        "Enhancing AI understanding of sarcasm...",
        "Deploying Mojo scripts for emotional support...",
        "Loading epic adventure maps..." "Compiling first lines of AI logic...",
        "Calibrating AI response timing...",
        "Testing AI with real-world scenarios...",
        "Uploading final Mojo modules...",
        "Finalizing AI personality traits...",
        "Conducting comprehensive Mojo system checks...",
        "Polishing AI's user interaction experience...",
        "Launching AI for beta testing...",
        "Implementing improvements and fixes...",
        "Reticulating pixelated landscapes..."
        "Compiling first quest objectives..."
        "Training noobs for battle readiness...",
        "Debugging the latest patch glitches...",
        "Optimizing frame rates for smooth gameplay...",
        "Executing legendary loot drops...",
        "Enhancing NPC AI for better dialogues...",
        "Deploying side quests for extra XP...",
        "Calibrating joystick sensitivity...",
        "Testing multiplayer connectivity...",
        "Uploading final character skins...",
        "Finalizing game world physics...",
        "Conducting last-minute gameplay tweaks...",
        "Polishing graphics for ultra realism...",
        "Launching open beta for player feedback...",
        "Collecting bug reports and praise...",
        "Implementing final patches and updates...",
        "Game release ready: Finally done. Achievement unlocked!",
    ]

    tasks = iter(c.split() for c in commands)
    my_env = os.environ.copy()
    my_env.update(env)
    task = subprocess.Popen(
        next(tasks), stdout=stdout, stderr=subprocess.STDOUT, env=my_env
    )
    tick = time.time() + random.randint(5, 20)

    # Gracefully exit if user kills the process (otherwise async shell commands will continue in the bg)
    def signal_handler(sig, frame):
        task.kill()
        sys.exit(0)

    clear_len = max(list(map(len, wait_strings))) + 8
    signal.signal(signal.SIGINT, signal_handler)

    total_tasks = len(commands)
    tasks_completed = 0
    while task is not None:
        if task.poll() is not None:
            try:
                task = subprocess.Popen(next(tasks), stdout=stdout)
                tasks_completed += 1
            except:
                task = None

        if time.time() > tick:
            tick += random.randint(5, 20)
            if print_progress:
                _wait_string_line(
                    wait_strings[random.randint(0, len(wait_strings)-1)],
                    clear_len,
                    100 * tasks_completed * 1.0 / total_tasks,
                )

        time.sleep(1)

    if print_progress:
        _wait_string_line("Done! ", clear_len, 100)
        print()


def _wait_string_line(line, clear_len, progress):
    print(
        f'\r{" "*(clear_len)}',
        end="",
        flush=True,
    )
    print(
        f"\r{line}" + f"[{progress:.0f}%]",
        end="",
        flush=True,
    )

def run(execute_cb):
    # Warm-up sample
    execute_cb()

    import time

    start = time.time()
    for i in range(NUM_ITERS):
        execute_cb()
        print(".", end="", flush=True)

    end = time.time()
    qps = NUM_ITERS * 1.0 / (end - start)
    print(f" QPS: {qps:.2f}")
    return qps
