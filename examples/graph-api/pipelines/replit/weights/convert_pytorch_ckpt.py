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
r"""
Usage:
  python3 convert_pytorch_ckpt.py \
    --input /path/to/replit/pytorch_model.bin \
    --output_dir /path/to/converted/files
"""

import os
import pathlib
import re
from argparse import ArgumentParser, FileType

import torch
from write_tensor import DType, write_tensor

_TORCH_DTYPE_MAP = {
    torch.float32: DType.f32,
    torch.float64: DType.f64,
    torch.float16: DType.f16,
    torch.bfloat16: DType.bf16,
    torch.int8: DType.si8,
    torch.uint8: DType.ui8,
    torch.int16: DType.si16,
    torch.int32: DType.si32,
    torch.int64: DType.si64,
    torch.bool: DType.kBool,
}


_ESCAPE_SUB = r"[^a-zA-Z0-9.\-_()]"


def convert(input_path: pathlib.Path, output_dir: pathlib.Path):
    weights = torch.load(input_path, weights_only=True)
    if not isinstance(weights, dict):
        # It's possible that weights could be Tensor/primitive type.
        raise TypeError(
            "This conversion script only supports converting state dicts."
        )
    for key, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor):
            raise NotImplementedError(
                "Primitives and nested dicts are not supported."
            )
        dtype = _TORCH_DTYPE_MAP[tensor.dtype]
        # Special case bfloat16 since it's not a numpy dtype
        if tensor.dtype == torch.bfloat16:
            # Convert to float32 since bfloat16 isn't supported on ARM yet.
            tensor_np = tensor.to(torch.float32).numpy()
            dtype = DType.f32
        else:
            tensor_np = tensor.numpy()
        with open(output_dir / re.sub(_ESCAPE_SUB, "_", key), "wb") as f:
            write_tensor(tensor_np, dtype, f)


def main():
    parser = ArgumentParser(
        description="Convert a Pytorch checkpoint to the Mojo Tensor format."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help=(
            "Path to PyTorch checkpoint file. The weights must be saved as a"
            " dictionary, such as from `torch.save(model.state_dict(), path)`."
        ),
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help=(
            "Path to directory that contains all converted Mojo Tensor files. "
            "The filenames will correspond to the weight keys."
        ),
        required=True,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Could not find input file {args.input}")
    if args.output_dir.exists():
        if args.output_dir.is_dir():
            if any(args.output_dir.iterdir()):
                raise ValueError(
                    f"Output directory {args.output_dir} already exists."
                )
        else:
            raise ValueError(f"{args.output_dir} already exists.")
    else:
        os.makedirs(args.output_dir)

    convert(args.input, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except:
        import pdb
        import traceback

        traceback.print_exc()
        pdb.post_mortem()
