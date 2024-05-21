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
@value
struct HyperParams:
    var batch_size: Int
    var seq_len: Int
    var n_heads: Int
    var causal: Bool
    var alibi: Bool
    var alibi_bias_max: Int
    var num_blocks: Int
    var vocab_size: Int
    var d_model: Int
    var kv_n_heads: Int


def get_default() -> HyperParams:
    return HyperParams(
        batch_size=1,
        seq_len=4096,
        n_heads=24,
        causal=True,
        alibi=True,
        alibi_bias_max=8,
        num_blocks=32,
        vocab_size=32768,
        d_model=3072,
        kv_n_heads=8,
    )
