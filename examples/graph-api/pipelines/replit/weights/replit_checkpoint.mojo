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
from pathlib import Path

from tensor import Tensor


trait Checkpoint(Movable):
    def __init__(inout self, path: Path):
        """Initializes the weights file from a path.

        Args:
            path: Filepath to the model's weights file.
        """
        ...

    def get[type: DType](self, key: String) -> Tensor[type]:
        """Returns a tensor for `key` at layer `layer_idx`, possibly seeking the file.

        `self` is `inout` here due to implementations that seek a file pointer.

        Args:
            key: Used to look up the tensor in the weights file.

        Returns:
            A tensor corresponding to `key` and `layer_idx` and containing a
            copy of data from the weights file.

        Raises:
            An error for invalid key arguments.
        """
        ...


struct ReplitCheckpoint(Checkpoint):
    """Reads from a directory containing serialized Mojo Tensors."""

    var model_path: Path

    def __init__(inout self, path: Path):
        """Initializes the weights from a path.

        Args:
            path: Path to model weights directory.
        """
        self.model_path = path

    fn __moveinit__(inout self, owned existing: Self):
        self.model_path = existing.model_path

    def get[type: DType](self, key: String) -> Tensor[type]:
        """Returns a tensor for `key` at layer `layer_idx`, possibly seeking the file.

        `self` is `inout` here due to implementations that seek a file pointer.

        Args:
            key: Used to look up the tensor in the weights file.

        Returns:
            A tensor corresponding to `key` and `layer_idx` and containing a
            copy of data from the weights file.

        Raises:
            An error for invalid key arguments.
        """
        path = self.model_path / key
        if not path.exists():
            raise "Could not load checkpoint tensor value. " + str(
                path
            ) + " does not exist."
        return Tensor[type].load(path)
