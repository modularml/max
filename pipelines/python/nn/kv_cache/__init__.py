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
from max.engine import InferenceSession
from max.driver import Device
from .cache_params import KVCacheLayout, KVCacheParams, KVCacheStrategy
from .naive_cache import NaiveKVCache
from .contiguous_cache import (
    ContiguousKVCacheType,
    ContiguousKVCacheCollectionType,
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    ContiguousKVCacheManager,
)
from .continuous_batching_cache import (
    ContinuousBatchingKVCacheCollectionType,
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheManager,
)
from .manager import KVCacheManager


def load_kv_manager(
    params: KVCacheParams,
    max_cache_batch_size: int,
    max_seq_len: int,
    num_layers: int,
    session: InferenceSession,
    device: Device,
) -> KVCacheManager:
    if params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return ContinuousBatchingKVCacheManager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            session=session,
            device=device,
        )
    elif params.cache_strategy == KVCacheStrategy.CONTIGUOUS:
        return ContiguousKVCacheManager(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            session=session,
            device=device,
        )
    else:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)
