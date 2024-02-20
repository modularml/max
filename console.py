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

try:
    from rich.console import Console
    from rich.table import Table

except ImportError:
    print("rich not found. Installing rich...")
    subprocess.run(["python3", "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table

ROOT = os.path.dirname(__file__)

def list_repositories():
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

def select_repository(repos):
    console = Console()
    table = Table(title="Select the Example to Run", highlight=True)
    table.add_column("Index", style="cyan", justify="center")
    table.add_column("MAX Engine Examples 🔥", style="magenta", justify="left")
    for index, (name, _) in enumerate(repos):
        table.add_row(str(index), name)

    console.print(table)
    selected_index = console.input("Enter the index of an example to run: ")
    return repos[int(selected_index)]


def run_repository(repo_name):
    repo_path = os.path.join(ROOT, repo_name)
    requirements_path = os.path.join(repo_path, "requirements.txt")
    run_script_path = os.path.join(repo_path, "run.sh")

    if os.path.exists(requirements_path):
        subprocess.run(["pip", "install", "-r", requirements_path])

    subprocess.run(["bash", run_script_path], cwd=repo_path)
    return


def main():
    repos = list_repositories()
    if not repos:
        print("No repositories found. Exiting ...")
        return

    console = Console()
    while True:
        _, selected_repo = select_repository(repos)
        console.print(f"Running {selected_repo} ...")
        run_repository(selected_repo)
        if console.input("Would you like to run another example? (y/n): ").lower() != "y":
            break


if __name__ == "__main__":
    main()
