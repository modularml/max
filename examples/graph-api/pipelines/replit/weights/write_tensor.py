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
import struct
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO, Sequence

import numpy as np

_SERIALIZATION_HEADER = b"\x93\xf0\x9f\x94\xa5\x93"
_SERIALIZATION_MAJOR_FORMAT = 0
_SERIALIZATION_MINOR_FORMAT = 1
_SPEC_SIZE_BYTES = 16  # sizeof[TensorSpec]()


class _Rep16:
    def __init__(self, shape: Sequence[int]):
        self.rank = len(shape)
        if self.rank > 6:
            raise ValueError("Can't create _Rep16 with rank > 6")
        for dim in shape:
            if dim > np.iinfo(np.int16).max:
                raise ValueError(
                    "Dimensions in _Rep16 can't be larger than the max value of"
                    f" Int16, got {dim}."
                )

        # Pad with 0s
        shape = list(shape)
        for _ in range(self.rank, 6):
            shape.append(0)

        self.shape = shape

    def to_bytes(self, dtype):
        s = b""
        # Add bytes for dims.
        for dim in self.shape:
            s += struct.pack("<h", dim)
        s += struct.pack("B", 0)  # Unused byte
        s += struct.pack("B", 0)  # Byte for KIND_16
        s += struct.pack("B", self.rank)  # Byte for rank
        s += struct.pack("B", dtype)  # Byte for auxillary (Tensor DType)
        return s


class _Rep32:
    def __init__(self, shape: Sequence[int]):
        self.rank = len(shape)
        if self.rank > 4:
            raise ValueError("Can't create _Rep32 with rank > 4")
        for dim in shape[:3]:
            if dim > np.iinfo(np.int32).max:
                raise ValueError(
                    "Dimensions in _Rep32 can't be larger than the max value of"
                    f" Int32, got {dim}."
                )
        if self.rank == 4:
            if shape[3] > np.iinfo(np.int8).max:
                raise ValueError(
                    "Last dimension in _Rep32 can't be larger than max value of"
                    f" Int8, got {dim}"
                )
        else:
            # Pad with 0s
            shape = list(shape)
            for _ in range(self.rank, 4):
                shape.append(0)
        self.shape = shape

    def to_bytes(self, dtype):
        s = b""
        # Add bytes for dims012.
        for dim in self.shape[:3]:
            s += struct.pack("<i", dim)
        s += struct.pack("B", self.shape[3])  # Byte for dims3
        s += struct.pack("B", 1)  # Byte for KIND_32
        s += struct.pack("B", self.rank)  # Byte for rank
        s += struct.pack("B", dtype)  # Byte for auxillary (Tensor DType)
        return s


# DType UInt8 values from Support/include/Support/ML/DType.h
mIsInteger = 1 << 7
mIsFloat = 1 << 6
mIsComplex = 1 << 5
mIsSigned = 1
kIntWidthShift = 1


class DType(Enum):
    invalid = 0
    si1 = (0 << kIntWidthShift) | mIsInteger | mIsSigned
    ui1 = (0 << kIntWidthShift) | mIsInteger
    si2 = (1 << kIntWidthShift) | mIsInteger | mIsSigned
    ui2 = (1 << kIntWidthShift) | mIsInteger
    si4 = (2 << kIntWidthShift) | mIsInteger | mIsSigned
    ui4 = (2 << kIntWidthShift) | mIsInteger
    si8 = (3 << kIntWidthShift) | mIsInteger | mIsSigned
    ui8 = (3 << kIntWidthShift) | mIsInteger
    si16 = (4 << kIntWidthShift) | mIsInteger | mIsSigned
    ui16 = (4 << kIntWidthShift) | mIsInteger
    si32 = (5 << kIntWidthShift) | mIsInteger | mIsSigned
    ui32 = (5 << kIntWidthShift) | mIsInteger
    si64 = (6 << kIntWidthShift) | mIsInteger | mIsSigned
    ui64 = (6 << kIntWidthShift) | mIsInteger
    si128 = (7 << kIntWidthShift) | mIsInteger | mIsSigned
    ui128 = (7 << kIntWidthShift) | mIsInteger

    f8 = 0 | mIsFloat
    f16 = 1 | mIsFloat
    f32 = 2 | mIsFloat
    f64 = 3 | mIsFloat
    f128 = 4 | mIsFloat

    bf16 = 5 | mIsFloat
    f24 = 6 | mIsFloat
    f80 = 7 | mIsFloat
    tf32 = 8 | mIsFloat

    kBool = 1


def write_tensor(tensor: np.ndarray, dtype: DType, f: BinaryIO):
    shape = tensor.shape

    try:
        tensor_spec = _Rep32(shape)
    except ValueError:
        try:
            tensor_spec = _Rep16(shape)
        except ValueError:
            raise ValueError(
                f"Serializing tensor with shape {shape} is not supported."
            )
    spec_bytes = tensor_spec.to_bytes(dtype.value)
    assert len(spec_bytes) == 16
    tensor_bytes = memoryview(tensor)
    f.write(_SERIALIZATION_HEADER)
    f.write(struct.pack("<i", _SERIALIZATION_MAJOR_FORMAT))
    f.write(struct.pack("<i", _SERIALIZATION_MINOR_FORMAT))
    f.write(struct.pack("<i", _SPEC_SIZE_BYTES))
    f.write(spec_bytes)
    f.write(tensor_bytes)
