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
from pathlib import cwd, Path
import sys

from ..configs.registry import ConfigRegistry, ConfigRegistryDict
from ..configs.parse_args import (
    OptionType,
    OptionTypeEnum,
    OptionValue,
    parse_args,
    register_pipeline_configs,
)


def get_replit_base_default_config() -> Dict[String, OptionValue]:
    default_config = Dict[String, OptionValue]()
    default_config["converted-weights-path"] = Path("")
    default_config["prompt"] = str('def hello():\n  print("hello world")')
    default_config["experimental-use-gpu"] = False
    default_config["quantization-encoding"] = str("float32")
    return default_config


@value
struct ReplitConfigRegistry(ConfigRegistry):
    """
    This struct holds a dictionary of Replit configs and their corresponding types.
    """

    var registry: ConfigRegistryDict

    def __init__(
        inout self,
        additional_pipeline_args: Optional[ConfigRegistryDict] = None,
    ):
        """
        This constructor instantiates Replit config keys and their
        corresponding types. An optional pipeline args dict is also available
        for use if a specific pipeline has additional arguments.
        """
        self.registry = ConfigRegistryDict()
        self.registry["converted-weights-path"] = OptionTypeEnum.PATH
        self.registry["prompt"] = OptionTypeEnum.STRING
        self.registry["max-length"] = OptionTypeEnum.INT
        self.registry["max-new-tokens"] = OptionTypeEnum.INT
        self.registry["experimental-use-gpu"] = OptionTypeEnum.BOOL
        self.registry["quantization-encoding"] = OptionTypeEnum.STRING
        if additional_pipeline_args:
            self.registry.update(additional_pipeline_args.value())

    def register(self, key: String, option_type: OptionType):
        self.registry[key] = option_type
