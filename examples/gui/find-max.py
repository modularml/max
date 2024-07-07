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

"""Find where max-engine is and install it"""

import glob
import os
import sys
from pathlib import Path

import toml


def install_max_dev():
    derived_path = os.getenv("MODULAR_DERIVED_PATH")

    if not derived_path:
        return

    pyproject_path = (
        Path(derived_path) / "build" / "SDK" / "lib" / "EngineAPI" / "python"
    )

    if not pyproject_path.is_dir():
        return

    # Load the TOML file
    with open("pyproject.toml", "r") as file:
        toml_data = toml.load(file)

    # Navigate to the target section
    tool_pixi_feature_dev = (
        toml_data.setdefault("tool", {})
        .setdefault("pixi", {})
        .setdefault("feature", {})
        .setdefault("dev", {})
        .setdefault("pypi-dependencies", {})
    )

    # Update the max-engine key
    max_engine = tool_pixi_feature_dev.setdefault("max-engine", {})
    max_engine.setdefault("path", str(pyproject_path))
    max_engine.setdefault("editable", True)

    # Navigate to the target section for environments.dev
    tool_pixi_environments_dev = (
        toml_data.setdefault("tool", {})
        .setdefault("pixi", {})
        .setdefault("environments", {})
        .setdefault("dev", {})
    )
    tool_pixi_environments_dev.setdefault("features", ["dev"])
    tool_pixi_environments_dev.setdefault("solve-group", "dev")

    # Write the updated data back to the TOML file
    with open("pyproject.toml", "w") as file:
        toml.dump(toml_data, file)

    print("TOML file has been updated successfully.")


def error(*error):
    for s in error:
        print(s, file=sys.stderr, end="", flush=True)
    print(flush=True)
    exit(1)


def add_wheels_to_project(wheels: Path, env: str):
    engine = ""
    lib = ""
    for path in wheels.glob("**/*"):
        if "cp311" in str(path):
            engine = path
        if "max_engine_libs" in str(path):
            lib = path

    if not engine or not lib:
        raise ImportError(
            "Could not find correct named wheels in install folder"
        )

    with open("pyproject.toml", "r") as file:
        toml_data = toml.load(file)

    project_optional_dependencies = toml_data.setdefault(
        "project", {}
    ).setdefault("optional-dependencies", {})
    project_optional_dependencies[env] = [
        f"max-engine @ file://{engine}",
        f"max-engine-libs @ file://{lib}",
    ]

    tool_pixi_environments_dev = (
        toml_data.setdefault("tool", {})
        .setdefault("pixi", {})
        .setdefault("environments", {})
        .setdefault(env, {})
    )
    tool_pixi_environments_dev.setdefault("features", [env])
    tool_pixi_environments_dev.setdefault("solve-group", env)

    with open("pyproject.toml", "w") as file:
        toml.dump(toml_data, file)


def install_max_release():
    modular_home = Path("~/.modular").expanduser()

    if "MODULAR_HOME" in os.environ:
        modular_home = Path(os.environ["MODULAR_HOME"])
        if not modular_home.is_dir():
            error(
                "Make sure MODULAR_HOME: '",
                modular_home,
                "' is an existing dir",
            )

    if not modular_home.is_dir():
        error("Install MAX first: https://modul.ar/install-max")

    pkg_path = modular_home / "pkg"
    stable_wheels = pkg_path / "packages.modular.com_max" / "wheels"
    nightly_wheels = pkg_path / "packages.modular.com_nightly_max" / "wheels"

    # If only nightly wheels exist, make them default and nightly feature
    if nightly_wheels.is_dir() and not stable_wheels.is_dir():
        add_wheels_to_project(nightly_wheels, "default")
        add_wheels_to_project(stable_wheels, "nightly")

    if stable_wheels.is_dir():
        add_wheels_to_project(stable_wheels, "default")

    if nightly_wheels.is_dir():
        add_wheels_to_project(nightly_wheels, "nightly")


install_max_dev()
install_max_release()
