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
"""A tokenizer that leverages the Hugging Face transformers library. This is
forked from the generic tokenizer library as the encode interfaces are different."""

from collections import Optional
from math import ceildiv
from memory import memcpy, UnsafePointer
from python import Python, PythonObject
from random import randint
from max.tensor import Tensor, TensorShape
from utils import Index

from .tokenizer import Tokenizer


struct AutoTokenizer(Tokenizer):
    """Wrapper around generic tokenizers loaded from the huggingface
    transformers library.

    WARN: There are some extra copies that happen in the naive form of this
    tokenizer, specifically when converting python list -> numpy tensor -> MAX Tensor.
    """

    var _transformers_module: PythonObject
    var _numpy_module: PythonObject
    var _tokenizer_handle: PythonObject
    var _py_builtins_handle: PythonObject
    var _prev_tokens: List[Int64]
    var _prev_decoded: String

    def __init__(out self, hf_tokenizer_name: String):
        self._transformers_module = Python.import_module("transformers")
        self._numpy_module = Python.import_module("numpy")
        self._tokenizer_handle = self._transformers_module.AutoTokenizer.from_pretrained(
            # Pad on the left so that the relevant output token remains as the last
            # element in the sequence regardless of how much padding is added.
            hf_tokenizer_name,
            padding_side="left",
        )
        # If the tokenizer does not set the padding token, then set it to EOS.
        # TODO(RUNP-242): Remove this once padding is no longer necessary.
        if not self._tokenizer_handle.pad_token:
            self._tokenizer_handle.pad_token = self._tokenizer_handle.eos_token
        self._py_builtins_handle = Python.import_module("builtins")
        self._prev_tokens = List[Int64]()
        self._prev_decoded = ""

    fn __moveinit__(out self, owned existing: Self):
        self._transformers_module = existing._transformers_module^
        self._numpy_module = existing._numpy_module^
        self._tokenizer_handle = existing._tokenizer_handle^
        self._py_builtins_handle = existing._py_builtins_handle^
        self._prev_tokens = existing._prev_tokens^
        self._prev_decoded = existing._prev_decoded^

    @always_inline
    @staticmethod
    def _numpy_data_pointer[
        type: DType
    ](numpy_array: PythonObject) -> UnsafePointer[Scalar[type]]:
        return numpy_array.__array_interface__["data"][0].unsafe_get_as_pointer[
            type
        ]()

    @always_inline
    @staticmethod
    def _memcpy_to_numpy(array: PythonObject, tokens: List[Int64]):
        dst = AutoTokenizer._numpy_data_pointer[DType.int64](array)
        memcpy(dst, tokens.unsafe_ptr(), len(tokens))

    @always_inline
    @staticmethod
    def _memcpy_from_numpy(array: PythonObject, tensor: Tensor):
        src = AutoTokenizer._numpy_data_pointer[tensor.type](array)
        dst = tensor._ptr
        length = tensor.num_elements()
        memcpy(dst, src, length)

    @staticmethod
    @always_inline
    def _numpy_to_tensor[type: DType](array: PythonObject) -> Tensor[type]:
        shape = List[Int]()
        array_shape = array.shape
        for dim in array_shape:
            shape.append(dim.__index__())
        out = Tensor[type](shape)
        AutoTokenizer._memcpy_from_numpy(array, out)
        return out^

    @always_inline
    def _list_of_string_to_py_list(
        self, string_list: List[String]
    ) -> PythonObject:
        input_string_py = self._py_builtins_handle.list()
        for i in range(len(string_list)):
            input_string_py.append(string_list[i])

        return input_string_py

    @always_inline
    def _shape_to_python_list(self, shape: TensorShape) -> PythonObject:
        python_list = self._py_builtins_handle.list()
        for i in range(shape.rank()):
            python_list.append(shape[i])
        return python_list^

    @always_inline
    def _get_np_dtype[type: DType](self) -> PythonObject:
        @parameter
        if type is DType.float32:
            return self._numpy_module.float32
        elif type is DType.int32:
            return self._numpy_module.int32
        elif type is DType.int64:
            return self._numpy_module.int64
        elif type is DType.uint8:
            return self._numpy_module.uint8

        raise Error("Unknown datatype")

    @always_inline
    def _tokens_to_numpy(self, tokens: List[Int64]) -> PythonObject:
        shape = self._shape_to_python_list(len(tokens))
        tokens_as_numpy = self._numpy_module.zeros(
            shape, self._get_np_dtype[DType.int64]()
        )
        self._memcpy_to_numpy(tokens_as_numpy, tokens)
        return tokens_as_numpy

    def is_end_of_text(self, val: Int64) -> Bool:
        return val == int(self._tokenizer_handle.eos_token_id)

    def encode(
        self,
        input_string: List[String],
        bos: Optional[String] = None,
        eos: Optional[String] = None,
        pad_to_multiple_of: Optional[Int] = None,
    ) -> (List[Int64], List[Int64]):
        input_string_py = self._list_of_string_to_py_list(input_string)

        if pad_to_multiple_of:
            # If padding is requested, invoke the tokenizer with padding=True
            # and pass the multiple_of value.
            # TODO(RUNP-242): Remove this as well as attn_mask result once
            #   padding is no longer necessary.
            tokenized_py = self._tokenizer_handle(
                input_string_py,
                padding=True,
                pad_to_multiple_of=pad_to_multiple_of.value(),
            )
        else:
            tokenized_py = self._tokenizer_handle(input_string_py)
        token_ids = AutoTokenizer._numpy_to_tensor[DType.int64](
            self._numpy_module.array(tokenized_py["input_ids"])
        )
        token_ids_list = List[Int64]()
        for i in range(token_ids.num_elements()):
            token_ids_list.append(token_ids._to_buffer()[i])
        _ = token_ids^

        attn_mask = AutoTokenizer._numpy_to_tensor[DType.int64](
            self._numpy_module.array(tokenized_py["attention_mask"])
        )
        attn_mask_list = List[Int64]()
        for i in range(attn_mask.num_elements()):
            attn_mask_list.append(attn_mask._to_buffer()[i])
        _ = attn_mask^

        return token_ids_list, attn_mask_list

    def decode(mut self, output_tokens: List[Int64]) -> String:
        """Decodes tokens using the autotokenizer and accounts for spaces."""

        # Attempt to produce correct output in a streaming setting.
        # Tokenizers decode differently depending on neighbouring tokens.
        # In particular, the sentencepiece BPE tokenizer removes prefix space
        # by default, so keep the previous decoded string and all previous
        # tokens around to correctly decode spaces.
        # Otherwise, when streaming one token at a time, no spaces are decoded.
        #
        # See for example:
        # https://github.com/huggingface/transformers/issues/22710).

        # Decode the full sequence.
        self._prev_tokens += output_tokens
        decoded = self._tokenizer_handle.decode(
            self._tokens_to_numpy(self._prev_tokens)
        )

        # Return the newly generated text.
        result = str(decoded)[len(self._prev_decoded) :]

        # Cache the full sequence for subsequent iterations.
        self._prev_decoded = str(decoded^)

        return result
