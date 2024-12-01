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
from collections import List, Dict
from pathlib import Path
from testing import assert_equal, assert_raises, assert_true

from pipelines.configs.parse_args import (
    OptionType,
    OptionTypeEnum,
    OptionValue,
    parse_args,
    register_pipeline_configs,
)
from pipelines.configs.registry import ConfigRegistry, ConfigRegistryDict


@value
struct DummyConfigRegistry(ConfigRegistry):
    """
    This struct holds a dictionary of dummy configs and their corresponding types.
    For testing purposes only.
    """

    var registry: ConfigRegistryDict

    def __init__(
        mut self,
        additional_pipeline_args: ConfigRegistryDict = ConfigRegistryDict(),
    ):
        self.registry = ConfigRegistryDict()
        self.registry["one"] = OptionTypeEnum.INT
        self.registry["two"] = OptionTypeEnum.PATH
        self.registry["three"] = OptionTypeEnum.PATH_LIST
        self.registry["four"] = OptionTypeEnum.STRING
        self.registry["five"] = OptionTypeEnum.FLOAT

        self.registry.update(additional_pipeline_args)

    def register(self, key: String, option_type: OptionType):
        self.registry[key] = option_type


def get_dummy_default_config() -> Dict[String, OptionValue]:
    default_config = Dict[String, OptionValue]()
    default_config["one"] = 1
    default_config["two"] = Path("")
    default_config["three"] = List[Path]()
    default_config["four"] = "four"
    default_config["five"] = 5.0
    return default_config


def test_register_pipeline_configs_no_defaults():
    config_registry = DummyConfigRegistry()
    test_args = List[String]("--one", "1", "--two", "", "--five", "5")

    # empty default config dict.
    default_configs = Dict[String, OptionValue]()
    curr_config = register_pipeline_configs(
        config_registry.registry,
        test_args,
        default_configs,
    )

    # Only 3 key val pairs are provided as CLI args.
    assert_equal(len(curr_config), 3)


def test_register_pipeline_configs_additional_configs():
    # Add additional arguments to config_registry
    config_registry_dict = ConfigRegistryDict()
    config_registry_dict["additional-flag"] = OptionTypeEnum.BOOL

    config_registry = DummyConfigRegistry(config_registry_dict)

    test_args = List[String](
        "--one", "1", "--two", "", "--five", "5", "--additional-flag"
    )
    # empty default config dict.
    default_configs = Dict[String, OptionValue]()
    curr_config = register_pipeline_configs(
        config_registry.registry,
        test_args,
        default_configs,
    )

    # Only 3 key val pairs are provided as CLI args.
    assert_equal(len(curr_config), 4)
    assert_true(curr_config.get("additional-flag"))


def test_register_pipeline_configs_overridden_configs():
    config_registry = DummyConfigRegistry()
    test_args = List[String](
        "--one", "1", "--four", "four-updated", "--five", "5"
    )

    # empty default config dict.
    default_configs = Dict[String, OptionValue]()
    default_configs["two"] = Path("two-default-val")
    default_configs["four"] = String("four-default-val")

    curr_config = register_pipeline_configs(
        config_registry.registry,
        test_args,
        default_configs,
    )

    # 3 key val pairs are provided as CLI args and the default dict also has an
    # additional key.
    assert_equal(len(curr_config), 4)

    # Since the "two" key hasn't been specified, we should already be falling back
    # on the default config "four-default-val"
    assert_equal(curr_config.get("two").value()[Path], Path("two-default-val"))

    # "four" is specified in both CLI arg and the default config, but CLI arg
    # takes precedence.
    four_updated = curr_config.get("four").value()[String]
    assert_equal(four_updated, String("four-updated"))


def test_register_pipeline_configs_missing_value_config():
    config_registry = DummyConfigRegistry()
    invalid_test_args = List[String]("--one")

    # empty default config dict.
    default_configs = Dict[String, OptionValue]()
    # double dashes
    with assert_raises(contains="Missing value for parameter"):
        curr_config = register_pipeline_configs(
            config_registry.registry,
            invalid_test_args,
            default_configs,
        )


def test_register_pipeline_configs_invalid_arg_config():
    config_registry = DummyConfigRegistry()
    invalid_test_args = List[String]("-one")

    # empty default config dict.
    default_configs = Dict[String, OptionValue]()
    # double dashes
    with assert_raises(contains="Valid arguments should start with --"):
        curr_config = register_pipeline_configs(
            config_registry.registry,
            invalid_test_args,
            default_configs,
        )


fn main() raises:
    test_register_pipeline_configs_no_defaults()
    test_register_pipeline_configs_additional_configs()
    test_register_pipeline_configs_overridden_configs()
    test_register_pipeline_configs_missing_value_config()
    test_register_pipeline_configs_invalid_arg_config()
