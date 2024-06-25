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

from max.engine import InferenceSession, Model, SessionOptions, TensorMap
from max._driver import cpu_device, cuda_device, Device
from tensor import Tensor, TensorShape
from utils import StaticTuple

from .model.replit import Replit
from .weights.replit_checkpoint import ReplitCheckpoint
from .weights.hyperparams import get_default
from ..samplers.token_sampler import TokenSampler
from ..samplers.weighted_sampler import WeightedSampler
from ..tokenizer import AutoTokenizer
from ..llama3.metrics import Metrics

# TODO: Expand this back out to 512 once MSDK-305 is fully resolved.
alias DEFAULT_MAX_SEQ_LEN = 33


@value
struct Config:
    """Configuration for token generation runtime options."""

    var converted_weights_path: Path
    var prompt: String
    var max_length: Optional[Int]
    var max_new_tokens: Optional[Int]
    var use_gpu: Bool
    var dtype: DType

    def __init__(
        inout self,
        /,
        converted_weights_path: Path = "",
        prompt: String = 'def hello():\n  print("hello world")',
        max_length: Optional[Int] = None,
        max_new_tokens: Optional[Int] = None,
        use_gpu: Bool = False,
        dtype: DType = DType.float32,
    ):
        self.converted_weights_path = converted_weights_path
        self.prompt = prompt
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.use_gpu = use_gpu
        self.dtype = dtype
        self.parse_args()

    def parse_args(inout self):
        args = sys.argv()
        raw_type = str("")

        @parameter
        def read_value(index: Int) -> StringRef:
            if index >= len(args):
                raise "missing value for parameter `" + str(
                    args[index - 1]
                ) + "`"
            return args[index]

        # Skip the run_pipeline.mojo and replit arguments.
        i = 2
        while i < len(args):
            if args[i] == "--converted_weights_path":
                self.converted_weights_path = Path(read_value(i + 1))
                i += 2
            elif args[i] == "--prompt":
                self.prompt = read_value(i + 1)
                i += 2
            elif args[i] == "--max_length":
                self.max_length = int(read_value(i + 1))
                i += 2
            elif args[i] == "--max_new_tokens":
                self.max_new_tokens = int(read_value(i + 1))
                i += 2
            elif args[i] == "--experimental-use-gpu":
                self.use_gpu = True
                i += 1
            elif args[i] == "--dtype":
                raw_type = read_value(i + 1)
                if not sys.info.is_x86() and raw_type == "bfloat16":
                    raise "bfloat16 is not supported for ARM architectures."
                if raw_type == "float32":
                    self.dtype = DType.float32
                elif raw_type == "bfloat16":
                    self.dtype = DType.bfloat16
                else:
                    raise "dtype must be 'bfloat16' or 'float32', got" + raw_type
                i += 2
            else:
                raise "unsupported CLI argument: " + String(args[i])

        print("dtype", self.dtype)
        if len(str(self.converted_weights_path)) == 0:
            if self.dtype == DType.float32:
                self.converted_weights_path = cwd().joinpath(
                    ".cache/replit/converted_float32"
                )
            else:  # DType.bfloat16
                self.converted_weights_path = cwd().joinpath(
                    ".cache/replit/converted_bfloat16"
                )
            if not self.converted_weights_path.exists():
                raise (
                    "Unable to find checkpoint at "
                    + str(self.converted_weights_path)
                    + ". Please run: setup.sh "
                    + raw_type
                )
            print("Using checkpoint at", self.converted_weights_path)


struct ReplitPipeline[dtype: DType]:
    """Code completion model based on Replit.

    Parameters:
        dtype: The DType of the weights and inputs to this model.
    """

    var _replit: Replit[ReplitCheckpoint, dtype]
    """Class that builds the Replit model graph."""

    var _compiled_model: Model
    """MAX Engine Model that contains the compiled graph and can be executed."""

    var _session: InferenceSession
    """MAX Engine session that holds holds and runs the model."""

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

    var _k_cache: Tensor[dtype]
    """Cache containing past computed attention keys."""

    var _v_cache: Tensor[dtype]
    """Cache containing past computed attention values."""

    var _next_token_tensor: Tensor[DType.int64]
    """ID of the last token generated by `ReplitPipeline.next_token()`, which
    will be used as the next input to the model."""

    var _cur_seq_len: Int
    """Length of the current sequence (including prompt)."""

    var _is_end_of_text: Bool
    """Whether text generation has reached an end-of-text token."""

    def __init__(
        inout self,
        checkpoint_file: Path,
        use_gpu: Bool = False,
        max_length: Optional[Int] = None,
        max_new_tokens: Optional[Int] = None,
    ):
        """Builds and compiles a Replit model to get ready for execution."""
        # Generate a graph that does a single forward pass of the replit model.
        print("Building model...")
        self._replit = Replit[ReplitCheckpoint, dtype](get_default())
        g = self._replit.build_graph(
            "replit",
            ReplitCheckpoint(checkpoint_file),
            with_attention_mask=True,
            use_cache=True,
        )

        # Load the graph into the session, which generates the MLIR and runs
        # optimization passes on it.
        print("Compiling...")
        var device = cuda_device() if use_gpu else cpu_device()
        var session_options = SessionOptions(device)
        self._session = InferenceSession(session_options)
        self._compiled_model = self._session.load(g)

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
        self._k_cache, self._v_cache = self._replit.create_empty_cache()
        self._next_token_tensor = Tensor[DType.int64]()
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
        self._k_cache, self._v_cache = self._replit.create_empty_cache()
        encoded_prompt = self._tokenizer.encode(List(prompt))

        self._next_token_tensor = Tensor(
            TensorShape(1, len(encoded_prompt)), encoded_prompt
        )
        self._cur_seq_len = len(encoded_prompt)
        self._max_seq_len = self._get_max_tokens(self._cur_seq_len)
        self._is_end_of_text = False
        return encoded_prompt.size

    def next_token(inout self) -> Optional[String]:
        """Generates the next token, or None if the end has been reached."""
        return self.next_token(WeightedSampler(0))

    def next_token[
        Sampler: TokenSampler
    ](inout self, sampler: Sampler) -> Optional[String]:
        """Generates the next token, or None if the end has been reached."""
        if self._is_end_of_text or self._max_seq_len - self._cur_seq_len <= 0:
            return None

        attention_mask = Tensor[DType.bool](
            TensorShape(1, self._cur_seq_len), True
        )
        results = execute(
            self._compiled_model,
            self._session,
            self._next_token_tensor,
            attention_mask,
            self._k_cache,
            self._v_cache,
        )
        logits = results.get[dtype]("output0")
        var token: SIMD[DType.int64, 1] = sampler.sample(logits).selected
        if self._tokenizer.is_end_of_text(token):
            self._is_end_of_text = True
            return None

        # Set up inputs for the next iteration.
        self._k_cache = results.get[dtype]("output1")
        self._v_cache = results.get[dtype]("output2")
        self._cur_seq_len += 1
        self._next_token_tensor = Tensor[DType.int64](TensorShape(1, 1), token)
        return self._tokenizer.decode(token)


def execute[
    dtype: DType
](
    model: Model,
    session: InferenceSession,
    tokens: Tensor[DType.int64],
    attention_mask: Tensor[DType.bool],
    k_cache: Tensor[dtype],
    v_cache: Tensor[dtype],
) -> TensorMap:
    input_map = session.new_tensor_map()
    input_map.borrow("input0", tokens)
    input_map.borrow("input1", attention_mask)
    input_map.borrow("input2", k_cache)
    input_map.borrow("input3", v_cache)
    result_map = model.execute(input_map)
    return result_map^


def dispatch[dtype: DType](config: Config):
    """Dispatches token generation for a model."""
    metrics = Metrics()
    metrics.begin_timing_startup()

    # Set up the Replit model prepare it for token generation.
    replit = ReplitPipeline[dtype](
        config.converted_weights_path,
        use_gpu=config.use_gpu,
        max_length=config.max_length,
        max_new_tokens=config.max_new_tokens,
    )
    metrics.end_timing_startup()

    input_string = config.prompt
    print("Running on input:", input_string)

    # Make sure newlines are properly encoded in the prompt.
    prompt = input_string.replace("\\n", "\n")

    # Run code generation.
    metrics.begin_timing_prompt()
    tokens_in_prompt = replit.reset(prompt)
    sampler = WeightedSampler(0.5)

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
    if not is_x86():
        dispatch[DType.float32](config)
    else:
        if config.dtype == DType.bfloat16:
            dispatch[DType.bfloat16](config)
        else:
            dispatch[DType.float32](config)
