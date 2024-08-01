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
from collections import Dict, Optional
import sys
from os import setenv

from max.engine import InferenceSession, Model
from max.graph.quantization import (
    BFloat16Encoding,
    Float32Encoding,
    QuantizationEncoding,
)
from max.tensor import TensorSpec
from max._driver import (
    Device,
    Tensor,
    AnyTensor,
    cuda_device,
    cpu_device,
    AnyMemory,
)
from pipelines.weights.gguf import GGUFFile

from .config import (
    ReplitConfigRegistry,
    get_replit_base_default_config,
    get_replit_model_url,
)
from .model.replit import Replit
from .weights.hyperparams import get_default

from ..configs.registry import ConfigRegistry, ConfigRegistryDict
from ..configs.parse_args import (
    OptionValue,
    parse_args,
    register_pipeline_configs,
)
from ..samplers.token_sampler import TokenSampler
from ..samplers.weighted_sampler import WeightedSampler
from ..tokenizer import AutoTokenizer
from ..llama3.metrics import Metrics
from ..weights.download import download_to_cache

alias DEFAULT_MAX_SEQ_LEN = 512


@value
struct Config:
    var config: Dict[String, OptionValue]
    var dtype: DType

    def __init__(
        inout self,
        additional_pipeline_args: Optional[ConfigRegistryDict] = None,
        additional_defaults: Optional[Dict[String, OptionValue]] = None,
    ):
        config_registry = ReplitConfigRegistry(additional_pipeline_args)
        default_configs = get_replit_base_default_config()
        if additional_defaults:
            default_configs.update(additional_defaults.value())

        self.config = register_pipeline_configs(
            config_registry.registry,
            parse_args(),
            default_configs,
        )

        # Finalize parsed arguments.
        self.dtype = DType.float32

        _raw_type = self.config["quantization-encoding"]
        raw_type = _raw_type[String]
        if not sys.info.is_x86() and raw_type == "bfloat16":
            raise "bfloat16 is not supported for ARM architectures."
        if raw_type == "float32":
            self.dtype = DType.float32
        elif raw_type == "bfloat16":
            self.dtype = DType.bfloat16
        else:
            raise "quantization-encoding must be 'bfloat16' or 'float32', got" + raw_type

        _model_path = self.config["model-path"]
        model_path = _model_path[Path]
        if not model_path:
            model_path = download_to_cache(get_replit_model_url(raw_type))
            print("Using checkpoint at", model_path)
            self.config["model-path"] = model_path
        if not model_path.exists():
            raise ("Unable to find checkpoint at " + str(model_path))

    def __contains__(self, key: String):
        return key in self.config

    fn get(inout self, key: String) raises -> OptionValue:
        """Returns an option value for `key` in the underlying config.

        Args:
            key: Key for the underlying config option.

        Returns:
            An OptionValue.

        Raises:
            An error for invalid key.
        """
        try:
            return self.config[key]
        except:
            raise "KeyError: " + key

    fn set(inout self, key: String, val: OptionValue):
        """Sets a new value for a given config key. This will overwrite the old
        value if the key is already present.

        Args:
            key: A string based key for the underlying config option.
            val: A new value for a key that already exist.
        """
        self.config[key] = val


struct ReplitPipeline[dtype: DType]:
    """Code completion model based on Replit.

    Parameters:
        dtype: The DType of the weights and inputs to this model.
    """

    var _replit: Replit[GGUFFile, dtype]
    """Class that builds the Replit model graph."""

    var _device: Device
    """Chosen device for execution."""

    var _run_on_gpu: Bool
    """Device chosen is gpu or not."""

    var _cpu_device: Device
    """An instance of cpu device. If chosen device is cpu this will
    be a copy of chosen device."""

    var _session: InferenceSession
    """MAX Engine session that holds and runs the model."""

    var _model: Model
    """Model compiled, initialized and ready for execution."""

    var _tokenizer: AutoTokenizer
    """Tokenizer for encoding/decoding the inputs and outputs."""

    # Token generation settings.
    var _max_length: Optional[Int]
    var _max_new_tokens: Optional[Int]

    # Attributes updated during generation.
    var _initial_prompt: String
    """Initial prompt user passed to `ReplitPipeline.reset()` method."""

    var _max_seq_len: Int
    """Maximum sequence length that will be generated by next_token(). This
    value includes the length of the inital prompt."""

    var _k_cache: AnyMemory
    """Cache containing past computed attention keys."""

    var _v_cache: AnyMemory
    """Cache containing past computed attention values."""

    var _next_token_tensor: AnyMemory
    """ID of the last token generated by `ReplitPipeline.next_token()`, which
    will be used as the next input to the model."""

    var _cur_seq_len: Int
    """Length of the current sequence (including prompt)."""

    var _is_end_of_text: Bool
    """Whether text generation has reached an end-of-text token."""

    def __init__(
        inout self,
        model_path: Path,
        use_gpu: Bool = False,
        max_length: Optional[Int] = None,
        max_new_tokens: Optional[Int] = None,
    ):
        """Builds and compiles a Replit model to get ready for execution."""
        # Generate a graph that does a single forward pass of the replit model.
        print("Building model...")
        self._replit = Replit[GGUFFile, dtype](get_default())
        model = GGUFFile(model_path)
        g = self._replit.build_graph(
            model,
            "replit",
            with_attention_mask=True,
            use_cache=True,
        )

        self._device = cuda_device() if use_gpu else cpu_device()
        self._run_on_gpu = use_gpu
        self._cpu_device = cpu_device() if use_gpu else self._device
        self._session = InferenceSession(self._device)

        # Compile and load the graph, which generates the MLIR and runs
        # optimization passes on it.
        print("Compiling...")
        self._model = self._session.load(g)

        # Set up tokenizer.
        var hf_model_name = "replit/replit-code-v1_5-3b"
        self._tokenizer = AutoTokenizer(hf_model_name)

        # Set default token generation options.
        self._max_length = None
        if max_length:
            self._max_length = max_length.value()
        self._max_new_tokens = None
        if max_new_tokens:
            self._max_new_tokens = max_new_tokens.value()

        # Initialize token generation attributes.
        self._initial_prompt = ""
        self._max_seq_len = 0
        kv_cache = self._replit.create_empty_cache(self._device)
        self._k_cache = kv_cache[0].take()
        self._v_cache = kv_cache[1].take()
        self._next_token_tensor = AnyTensor()
        self._cur_seq_len = 0
        self._is_end_of_text = True

    def _get_max_tokens(self, prompt_len: Int) -> Int:
        """Returns the max sequence length to generate (including the prompt).
        """
        if self._max_length:
            if self._max_new_tokens:
                return min(
                    self._max_new_tokens.value() + prompt_len,
                    self._max_length.value(),
                )
            else:
                return self._max_length.value()
        elif self._max_new_tokens:
            return self._max_new_tokens.value() + prompt_len
        else:
            return DEFAULT_MAX_SEQ_LEN

    def reset(inout self, prompt: String) -> Int:
        """Resets the prompt and model state."""
        self._initial_prompt = prompt
        self._max_seq_len = self._get_max_tokens(len(prompt))
        kv_cache = self._replit.create_empty_cache(self._device)
        self._k_cache = kv_cache[0].take()
        self._v_cache = kv_cache[1].take()

        encoded_prompt = self._tokenizer.encode(List(prompt))
        next_token_tensor = Tensor[DType.int64, 2](
            (1, len(encoded_prompt)), self._cpu_device
        )
        for i in range(len(encoded_prompt)):
            next_token_tensor[0, i] = encoded_prompt[i]
        self._set_next_token_tensor(next_token_tensor)

        self._cur_seq_len = len(encoded_prompt)
        self._max_seq_len = self._get_max_tokens(self._cur_seq_len)
        self._is_end_of_text = False
        return encoded_prompt.size

    def next_token(inout self) -> Optional[String]:
        """Generates the next token, or None if the end has been reached."""
        return self.next_token(WeightedSampler(0))

    def _set_next_token_tensor(inout self, owned next_token_tensor: AnyTensor):
        """Set the given value as next token tensor. If the chosen
        device is gpu, value will be copied over to the device."""

        self._next_token_tensor = next_token_tensor^

    def _get_attention_mask(self) -> AnyTensor:
        """Generates attention mask for current input sequence.
        Result is placed on the chosen device.
        """

        attention_mask_tensor = Tensor[DType.bool, 2](
            (1, self._cur_seq_len), self._cpu_device
        )
        for i in range(self._cur_seq_len):
            attention_mask_tensor[0, i] = True

        return attention_mask_tensor

    def next_token[
        Sampler: TokenSampler
    ](inout self, sampler: Sampler) -> Optional[String]:
        """Generates the next token, or None if the end has been reached."""
        if self._is_end_of_text or self._max_seq_len - self._cur_seq_len <= 0:
            return None

        results = self._model._execute(
            self._next_token_tensor.take(),
            self._get_attention_mask(),
            self._k_cache.take(),
            self._v_cache.take(),
        )

        output = results[0].take()
        self._k_cache = results[1].take()
        self._v_cache = results[2].take()

        logits = output.to_device_tensor()
        if self._run_on_gpu:
            logits = logits.copy_to(self._cpu_device)
        var token: Int64 = sampler.sample(logits.to_tensor[dtype, 2]()).selected
        if self._tokenizer.is_end_of_text(token):
            self._is_end_of_text = True
            return None
        self._cur_seq_len += 1

        next_token_tensor = Tensor[DType.int64, 2]((1, 1), self._cpu_device)
        next_token_tensor[0, 0] = token
        self._set_next_token_tensor(next_token_tensor)

        return self._tokenizer.decode(token)


def dispatch[dtype: DType](config: Config):
    """Dispatches token generation for a model."""
    metrics = Metrics()
    metrics.begin_timing_startup()

    # Set up the Replit model prepare it for token generation.
    var max_length: Optional[Int] = None
    if "max-length" in config:
        max_length = config.get("max-length")[Int]
    var max_new_tokens: Optional[Int] = None
    if "max-new-tokens" in config:
        max_new_tokens = config.get("max-new-tokens")[Int]
    replit = ReplitPipeline[dtype](
        config.get("model-path")[Path],
        use_gpu=config.get("experimental-use-gpu")[Bool],
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    metrics.end_timing_startup()

    input_string = config.get("prompt")[String]
    print("Running on input:", input_string)

    # Make sure newlines are properly encoded in the prompt.
    prompt = input_string.replace("\\n", "\n")

    # Run code generation.
    metrics.begin_timing_prompt()
    sampler = WeightedSampler(
        config.get("temperature")[Float64].cast[DType.float32](),
        config.get("min-p")[Float64].cast[DType.float32](),
    )

    # If a pipeline warmup is needed, run a single token completion after the
    # prompt, get a token after that, and reset.
    if config.get("warmup-pipeline")[Bool]:
        print("Warming up pipeline...")
        metrics.begin_timing_warmup()
        _ = replit.reset(prompt)
        _ = replit.next_token(sampler)
        _ = replit.next_token(sampler)
        metrics.end_timing_warmup()

    tokens_in_prompt = replit.reset(prompt)
    metrics.set_tokens_in_prompt(tokens_in_prompt)

    print("Output:")
    metrics.begin_timing_generation()
    while True:
        s = replit.next_token(sampler)
        if not s:
            break
        metrics.new_token()
        print(s.value(), end="")
    metrics.end_timing()
    print()
    metrics.print()


def replit_run():
    config = Config()

    @parameter
    if not sys.info.is_x86():
        dispatch[DType.float32](config)
    else:
        encoding = config.get("quantization-encoding")[String]
        if encoding == BFloat16Encoding.id():
            dispatch[DType.bfloat16](config)
        elif encoding == Float32Encoding.id():
            dispatch[DType.float32](config)
        else:
            raise "--quantization-encoding must be 'bfloat16' or 'float32', got" + encoding
