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

"""Llama 3.2 Transformer Vision Self Attention layer."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
)
from nn import AttentionWithRopeQKV


@dataclass
class SelfSdpaAttention(AttentionWithRopeQKV):
    def __call__(
        self,
        x: TensorValueLike,
        kv_collection: ContinuousBatchingKVCacheCollectionType,
        **kwargs,
    ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]:
        # This layer requires inputs to have a rank of 2, so we reshape and then
        # reuse Attention with Rope for it.
        batch_size, seq_len, hidden_dim = x.shape
        x = x.reshape((batch_size * seq_len, hidden_dim))

        x, kv_collection = super().__call__(x, kv_collection, **kwargs)

        x = x.reshape((batch_size, seq_len, hidden_dim))
        return x, kv_collection  # type: ignore
