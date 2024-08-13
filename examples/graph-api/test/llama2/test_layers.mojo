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
from max.graph import Dim, Graph, Symbol, TensorType, Type
from max.graph._testing import (
    assert_tensors_almost_equal,
    execute_binary,
    execute_nullary,
    execute_n_args,
    execute_unary,
)
from max.tensor import Tensor, TensorShape

from pipelines.nn import (
    Attention,
    Embedding,
    FeedForward,
    RMSNorm,
    Transformer,
    TransformerBlock,
)
from pipelines.nn.attention import expand_attention_mask


fn test_freqs_cis() raises:
    var g = Graph(List[Type]())
    var dummy = Transformer(
        dim=2,
        n_heads=1,
        embedding=Embedding(g.constant(Tensor[DType.float32](10, 10))),
        layers=List[TransformerBlock[DType.float32]](),
        norm=RMSNorm(1e-5, g.constant(Tensor[DType.float32](2))),
        output=g.constant(Tensor[DType.float32](10, 10)),
        theta=10000.0,
    )
    _ = g.output(
        dummy.freqs_cis(
            g.scalar[DType.int64](0), g.scalar[DType.int64](8), Dim("seq_len")
        )
    )

    # This test vector was produced from the FB Llama repo:
    # https://github.com/facebookresearch/llama/blob/4835a30a1cc98bc6894c17938178c8fc2f694a5d/llama/model.py#L80
    var expected = Tensor[DType.float32](
        TensorShape(8, 1, 2),
        1.0000,
        0.0000,
        0.5403,
        0.8415,
        -0.4161,
        0.9093,
        -0.9900,
        0.1411,
        -0.6536,
        -0.7568,
        0.2837,
        -0.9589,
        0.9602,
        -0.2794,
        0.7539,
        0.6570,
    )

    var actual = execute_nullary(g)

    assert_tensors_almost_equal(actual, expected, atol=1e-5, rtol=1e-3)


fn test_feed_forward() raises:
    alias dim = 2
    alias hidden_dim = 2

    var g = Graph(TensorType(DType.float32, "batch", "seq_len", dim))
    var layer = FeedForward(
        w1=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, hidden_dim), 0.5641, -1.1172, 0.4875, -1.1583
            )
        ),
        w2=g.constant(
            Tensor[DType.float32](
                TensorShape(hidden_dim, dim), 0.5355, -0.6487, -0.9487, 0.1838
            )
        ),
        w3=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, hidden_dim), -0.6765, -0.4643, 0.7103, 0.2860
            )
        ),
    )
    _ = g.output(layer(g[0]))

    var input = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -1.2620,
        -2.0678,
        -1.6634,
        1.3036,
        -0.0088,
        -1.1315,
        1.1287,
        1.7699,
    )
    var expected = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        0.1053,
        -0.1079,
        -0.3632,
        0.2142,
        0.4025,
        -0.1661,
        0.3220,
        -0.3921,
    )

    var actual = execute_unary(g, input)

    assert_tensors_almost_equal(actual, expected, atol=1e-5, rtol=1e-3)


fn test_rms_norm() raises:
    alias dim = 2

    var g = Graph(TensorType(DType.float32, "batch", "seq_len", dim))
    var layer = RMSNorm(
        1e-5,
        g.constant(Tensor[DType.float32](TensorShape(dim), 1.0476, -0.3264)),
    )
    _ = g.output(layer(g[0]))

    var input = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -1.2620,
        -2.0678,
        -1.6634,
        1.3036,
        -0.0088,
        -1.1315,
        1.1287,
        1.7699,
    )
    var expected = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -0.7718,
        0.3940,
        -1.1661,
        -0.2847,
        -0.0115,
        0.4616,
        0.7966,
        -0.3892,
    )

    var actual = execute_unary(g, input)

    assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_attention_mask() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int64, "prev_seq_len", "seq_len"),
            TensorType(DType.bool, 1, "full_seq_len"),
        )
    )
    var shape = g[0].shape()
    var prev_seq_len = shape[0]
    var seq_len = shape[1]
    _ = g.output(
        expand_attention_mask(g, g[1], prev_seq_len, seq_len, DType.float32)
    )

    var input = Tensor[DType.int64](TensorShape(0, 2), 0)
    var input_mask = Tensor[DType.bool](TensorShape(1, 2), True)
    var actual = execute_binary[outtype = DType.float32](g, input, input_mask)

    assert_tensors_almost_equal(
        actual,
        Tensor[DType.float32](
            TensorShape(1, 2, 2),
            0,
            -10000,
            0,
            0,
        ),
    )

    input = Tensor[DType.int64](TensorShape(2, 1), 0)
    input_mask = Tensor[DType.bool](TensorShape(1, 3), True)
    actual = execute_binary[outtype = DType.float32](g, input, input_mask)

    assert_tensors_almost_equal(
        actual,
        Tensor[DType.float32](TensorShape(1, 1, 3), 0, 0, 0),
    )

    # Test the uncommon case with non-zero prev_seq_len and curr_seq_len > 1.
    input = Tensor[DType.int64](TensorShape(1, 2), 0)
    input_mask = Tensor[DType.bool](TensorShape(1, 3), True)
    actual = execute_binary[outtype = DType.float32](g, input, input_mask)

    assert_tensors_almost_equal(
        actual,
        Tensor[DType.float32](TensorShape(1, 2, 3), 0, 0, -10000, 0, 0, 0),
    )


fn test_attention() raises:
    alias dim = 2
    alias n_heads = 1
    alias n_kv_heads = 1
    alias head_dim = 2

    var batch = "batch"
    var seq_len = "seq_len"
    var prev_seq_len = "prev_seq_len"
    var full_seq_len = "full_seq_len"
    var g = Graph(
        List[Type](
            TensorType(DType.float32, batch, seq_len, dim),
            TensorType(DType.float32, seq_len, 1, 2),
            TensorType(DType.bool, batch, full_seq_len),
            TensorType(
                DType.float32,
                prev_seq_len,
                1,
                batch,
                n_kv_heads,
                head_dim,
            ),
            TensorType(
                DType.float32,
                prev_seq_len,
                1,
                batch,
                n_kv_heads,
                head_dim,
            ),
        )
    )
    var layer = Attention(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        use_custom_attention=False,
        wq=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_heads * head_dim),
                0.3256,
                -0.4062,
                -1.8786,
                -0.4507,
            )
        ),
        wk=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_kv_heads * head_dim),
                0.6694,
                0.8910,
                -0.7980,
                0.9103,
            )
        ),
        wv=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_kv_heads * head_dim),
                0.5933,
                -0.0971,
                1.0371,
                0.0469,
            )
        ),
        wo=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_heads * head_dim),
                0.0713,
                0.0103,
                0.3269,
                -0.0694,
            )
        ),
    )
    var out = layer(g[0], g[1], g[2], g[3], g[4])
    _ = g.output(List[Symbol](out[0], out[1], out[2]))

    var input = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -1.2620,
        -2.0678,
        -1.6634,
        1.3036,
        -0.0088,
        -1.1315,
        1.1287,
        1.7699,
    )
    var attn_mask = Tensor[DType.bool](TensorShape(2, 2), True)
    # This is a complex tensor of shape (2, 1); we use the last dim as
    # (real, imag).
    # These values generated from the correct freqs_cis given Llama hyperparams.
    var freq_cis = Tensor[DType.float32](
        TensorShape(2, 1, 2), 1.0, 0.0, 0.5403, 0.8415
    )
    var k_cache = Tensor[DType.float32](
        TensorShape(0, 1, 2, n_kv_heads, head_dim)
    )
    var v_cache = Tensor[DType.float32](
        TensorShape(0, 1, 2, n_kv_heads, head_dim)
    )
    var expected_tokens = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -0.1979,
        -0.0316,
        -0.0312,
        -0.0204,
        -0.1011,
        -0.0085,
        -0.0874,
        -0.0067,
    )
    var expected_k_cache = Tensor[DType.float32](
        TensorShape(2, 2, n_kv_heads, head_dim),
        0.8053,
        -3.0068,
        -0.9151,
        -1.9720,
        0.8970,
        -1.0378,
        -2.5569,
        0.8611,
    )
    var expected_v_cache = Tensor[DType.float32](
        TensorShape(2, 2, n_kv_heads, head_dim),
        -2.8933,
        0.0256,
        0.3651,
        0.2227,
        -1.1787,
        -0.0522,
        2.5052,
        -0.0266,
    )

    var actuals = execute_n_args(
        g, input, freq_cis, attn_mask, k_cache, v_cache
    )

    assert_tensors_almost_equal(
        actuals.get[DType.float32]("output0"),
        expected_tokens,
        atol=1e-4,
        rtol=1e-4,
    )
    assert_tensors_almost_equal(
        actuals.get[DType.float32]("output1"),
        expected_k_cache,
        atol=1e-4,
        rtol=1e-4,
    )
    assert_tensors_almost_equal(
        actuals.get[DType.float32]("output2"),
        expected_v_cache,
        atol=1e-4,
        rtol=1e-4,
    )


fn test_transformer_block() raises:
    alias dim = 2
    alias n_heads = 1
    alias n_kv_heads = 1
    alias head_dim = 2
    alias hidden_dim = 2

    var batch = "batch"
    var seq_len = "seq_len"
    var prev_seq_len = "prev_seq_len"
    var full_seq_len = "full_seq_len"
    var g = Graph(
        List[Type](
            TensorType(DType.float32, batch, seq_len, dim),
            TensorType(DType.float32, seq_len, 1, 2),
            TensorType(DType.bool, batch, full_seq_len),
            TensorType(
                DType.float32,
                prev_seq_len,
                1,
                batch,
                n_kv_heads,
                head_dim,
            ),
            TensorType(
                DType.float32,
                prev_seq_len,
                1,
                batch,
                n_kv_heads,
                head_dim,
            ),
        )
    )
    var attention = Attention(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        use_custom_attention=False,
        wq=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_heads * head_dim),
                0.3256,
                -0.4062,
                -1.8786,
                -0.4507,
            )
        ),
        wk=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_kv_heads * head_dim),
                0.6694,
                0.8910,
                -0.7980,
                0.9103,
            )
        ),
        wv=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_kv_heads * head_dim),
                0.5933,
                -0.0971,
                1.0371,
                0.0469,
            )
        ),
        wo=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, n_heads * head_dim),
                0.0713,
                0.0103,
                0.3269,
                -0.0694,
            )
        ),
    )
    var feed_forward = FeedForward(
        w1=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, hidden_dim), 0.5641, -1.1172, 0.4875, -1.1583
            )
        ),
        w2=g.constant(
            Tensor[DType.float32](
                TensorShape(hidden_dim, dim), 0.5355, -0.6487, -0.9487, 0.1838
            )
        ),
        w3=g.constant(
            Tensor[DType.float32](
                TensorShape(dim, hidden_dim), -0.6765, -0.4643, 0.7103, 0.2860
            )
        ),
    )
    var attention_norm = RMSNorm(
        1e-5,
        g.constant(Tensor[DType.float32](TensorShape(dim), -0.0766, 0.6322)),
    )
    var ffn_norm = RMSNorm(
        1e-5,
        g.constant(Tensor[DType.float32](TensorShape(dim), -1.0754, -1.1960)),
    )
    var layer = TransformerBlock(
        attention=attention,
        feed_forward=feed_forward,
        attention_norm=attention_norm,
        ffn_norm=ffn_norm,
    )
    var out = layer(g[0], g[1], g[2], g[3], g[4])
    _ = g.output(List[Symbol](out[0], out[1], out[2]))

    var input = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -1.2620,
        -2.0678,
        -1.6634,
        1.3036,
        -0.0088,
        -1.1315,
        1.1287,
        1.7699,
    )
    var attn_mask = Tensor[DType.bool](TensorShape(2, 2), True)

    # This is a complex tensor of shape (2, 1); we use the last dim as
    # (real, imag).
    # These values generated from the correct freqs_cis given Llama hyperparams.
    var freq_cis = Tensor[DType.float32](
        TensorShape(2, 1, 2), 1.0, 0.0, 0.5403, 0.8415
    )
    var k_cache = Tensor[DType.float32](
        TensorShape(0, 1, 2, n_kv_heads, head_dim)
    )
    var v_cache = Tensor[DType.float32](
        TensorShape(0, 1, 2, n_kv_heads, head_dim)
    )
    var expected_tokens = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -1.1102,
        -2.3339,
        -1.8064,
        1.4070,
        0.3818,
        -1.6129,
        1.2590,
        1.6723,
    )
    var expected_k_cache = Tensor[DType.float32](
        TensorShape(2, 2, n_kv_heads, head_dim),
        0.6468,
        -0.6444,
        -0.6933,
        -0.0100,
        0.7140,
        -0.8131,
        -0.8799,
        -0.1963,
    )
    var expected_v_cache = Tensor[DType.float32](
        TensorShape(2, 2, n_kv_heads, head_dim),
        -0.7580,
        -0.0413,
        0.6225,
        0.0176,
        -0.9267,
        -0.0420,
        0.7472,
        0.0410,
    )

    var actuals = execute_n_args(
        g, input, freq_cis, attn_mask, k_cache, v_cache
    )

    assert_tensors_almost_equal(
        actuals.get[DType.float32]("output0"),
        expected_tokens,
        atol=1e-4,
        rtol=1e-4,
    )
    assert_tensors_almost_equal(
        actuals.get[DType.float32]("output1"),
        expected_k_cache,
        atol=1e-4,
        rtol=1e-4,
    )
    assert_tensors_almost_equal(
        actuals.get[DType.float32]("output2"),
        expected_v_cache,
        atol=1e-4,
        rtol=1e-4,
    )


def main():
    test_freqs_cis()
    test_feed_forward()
    test_rms_norm()
    test_attention_mask()
    test_attention()
    test_transformer_block()
