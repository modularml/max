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
from collections import Dict, Optional
from pathlib import cwd, Path
import sys

from max.graph.quantization import (
    BFloat16Encoding,
    Float32Encoding,
    QuantizationEncoding,
)
from ..configs.common import check_url_exists
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
    default_config["model-path"] = Path("")
    default_config["custom-ops-path"] = List[Path]()
    default_config["prompt"] = str('def hello():\n  print("hello world")')
    default_config["experimental-use-gpu"] = False
    default_config["quantization-encoding"] = str("float32")
    default_config["temperature"] = 1.0
    default_config["min-p"] = 0.05
    default_config["num-warmups"] = 1
    default_config["max-batch-size"] = 1
    default_config["max-length"] = 512
    default_config["custom-ops-path"] = Path("")
    default_config["mef-use-or-gen-path"] = str("")
    return default_config


# fmt: off
def get_replit_model_url(encoding: String) -> String:
    urls = Dict[String, String]()
    urls[BFloat16Encoding.id()] = "https://huggingface.co/tzhenghao/replit-code-v1_5-3b-bf16.gguf/resolve/main/replit-code-v1_5-3b-bf16.gguf"
    urls[Float32Encoding.id()] = "https://huggingface.co/tzhenghao/replit-code-v1_5-3b-f32.gguf/resolve/main/replit-code-v1_5-3b-f32.gguf"
    check_url_exists(urls, encoding)
    return urls[encoding]
# fmt: on


@value
struct ReplitConfigRegistry(ConfigRegistry):
    """
    This struct holds a dictionary of Replit configs and their corresponding types.
    """

    var registry: ConfigRegistryDict

    def __init__(
        mut self,
        additional_pipeline_args: Optional[ConfigRegistryDict] = None,
    ):
        """
        This constructor instantiates Replit config keys and their
        corresponding types. An optional pipeline args dict is also available
        for use if a specific pipeline has additional arguments.
        """
        self.registry = ConfigRegistryDict()
        self.registry["model-path"] = OptionTypeEnum.PATH
        self.registry["custom-ops-path"] = OptionTypeEnum.PATH_LIST
        self.registry["prompt"] = OptionTypeEnum.STRING
        self.registry["max-length"] = OptionTypeEnum.INT
        self.registry["max-new-tokens"] = OptionTypeEnum.INT
        self.registry["experimental-use-gpu"] = OptionTypeEnum.BOOL
        self.registry["quantization-encoding"] = OptionTypeEnum.STRING
        self.registry["temperature"] = OptionTypeEnum.FLOAT
        self.registry["min-p"] = OptionTypeEnum.FLOAT
        self.registry["num-warmups"] = OptionTypeEnum.INT
        self.registry["pad-to-multiple-of"] = OptionTypeEnum.INT
        self.registry["max-batch-size"] = OptionTypeEnum.INT
        self.registry["mef-use-or-gen-path"] = OptionTypeEnum.STRING
        if additional_pipeline_args:
            self.registry.update(additional_pipeline_args.value())

    def register(self, key: String, option_type: OptionType):
        self.registry[key] = option_type
