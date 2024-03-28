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

from enum import Enum
import os
from pathlib import Path
import subprocess
from typing import Optional, List, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Confirm

except ImportError:
    print("rich not found. Installing rich...")
    subprocess.run(["python3", "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Confirm

ROOT = Path(os.path.dirname(__file__))
RETRIES = 10
EXCEEDED_RETRY_ERROR = (
    "Exceeded the number of retries. Please re-run the console again and follow"
    " the prompt."
)


def list_repositories() -> List[Tuple[str, str]]:
    repos = []
    for top_repo in os.listdir(ROOT):
        if "notebook" in top_repo or "console.py" in top_repo:
            continue

        repo = os.path.join(ROOT, top_repo)
        for r in os.listdir(repo):
            full_path = os.path.join(repo, r)
            if os.path.isdir(full_path) and os.path.exists(
                os.path.join(full_path, "run.sh")
            ):
                repos.append((os.path.join(top_repo, r), full_path))

    repos = sorted(repos, key=lambda x: x[0])
    return repos


class InputState(Enum):
    PROMPT_INPUT = 0
    VALIDATE_INPUT = 1
    CHECK_RANGE = 2


def prompt_validation(
    console: Console, retries: int, repos: List[Tuple[str, str]]
) -> Optional[Tuple[str, str]]:
    state = InputState.PROMPT_INPUT
    selected_index = None
    n_repos = len(repos)
    while retries > 0:
        if state == InputState.PROMPT_INPUT:
            selected_index = console.input(
                "Enter the index of an example to run: "
            )
            state = InputState.VALIDATE_INPUT
        elif state == InputState.VALIDATE_INPUT:
            if not selected_index.strip():
                selected_index = console.input(
                    f"Please enter an index between {0}-{n_repos - 1}: "
                )
                retries -= 1
                if retries <= 0:
                    console.print(EXCEEDED_RETRY_ERROR, style="red")
                    return None
                else:
                    continue

            try:
                selected_index = int(selected_index)
                state = InputState.CHECK_RANGE
            except ValueError:
                selected_index = console.input(
                    f"The index must be an integer between {0}-{n_repos - 1}: "
                )
                retries -= 1
                if retries <= 0:
                    console.print(EXCEEDED_RETRY_ERROR, style="red")
                    return None
                else:
                    state = InputState.VALIDATE_INPUT
                    continue

        elif state == InputState.CHECK_RANGE:
            if 0 <= selected_index < n_repos:
                return repos[selected_index]
            else:
                selected_index = console.input(
                    f"Please enter an index between {0}-{n_repos - 1}: "
                )
                retries -= 1
                if retries <= 0:
                    console.print(EXCEEDED_RETRY_ERROR, style="red")
                    return None
                else:
                    state = InputState.VALIDATE_INPUT
                    continue

    console.print(EXCEEDED_RETRY_ERROR)
    return None


def select_repository(
    console: Console,
    repos: List[Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    table = Table(title="Select the Example to Run", highlight=True)
    table.add_column("Index", style="cyan", justify="center")
    table.add_column(
        "MAX Engine 🏎️  Examples 🔥", style="magenta", justify="left"
    )
    for index, (name, _) in enumerate(repos):
        table.add_row(str(index), name)

    console.print(table)
    selected_repo = prompt_validation(console, RETRIES, repos)
    if selected_repo is None:
        return

    return selected_repo


def run_repository(repo_name: Tuple[str, str]) -> None:
    repo_path = os.path.join(ROOT, repo_name)
    requirements_path = os.path.join(repo_path, "requirements.txt")
    run_script_path = os.path.join(repo_path, "run.sh")

    if os.path.exists(requirements_path):
        subprocess.run(["pip", "install", "-r", requirements_path])

    subprocess.run(["bash", run_script_path], cwd=repo_path)
    return


def main():
    repos = list_repositories()
    console = Console()
    if not repos:
        console.print("No repositories found. Exiting!", style="red")
        return

    exit_ = False
    while not exit_:
        console.print("\n")
        selected = select_repository(console, repos)
        if selected is None:
            return

        _, selected_repo = selected
        console.print(f"Running {selected_repo} ...")
        run_repository(selected_repo)
        another = Confirm.ask("Would you like to run another example?")
        if not another:
            exit_ = True
            console.print(
                "Thanks for trying the examples! Bye 👋", style="green"
            )
            break
        else:
            console.print("Here is the example table again \n", style="green")


if __name__ == "__main__":
    main()
