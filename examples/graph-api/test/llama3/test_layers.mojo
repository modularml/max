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

from collections import List, Dict, Optional
from pathlib import Path
from max.tensor import Tensor, TensorShape

from max.graph import _testing, Type, Dim, Graph, TensorType, Symbol, ops

from pipelines.llama3.model import Llama3
from pipelines.nn.attention import expand_attention_mask, rope
from pipelines.nn import (
    Attention,
    Embedding,
    FeedForward,
    RMSNorm,
    Transformer,
    TransformerBlock,
)

from pipelines.weights.loadable_model import LoadableModel, LlamaHParams


# fmt: off
@value
struct NanoLlama(LoadableModel):
    var weights: Dict[String, Tensor[DType.float32]]

    fn __init__(out self, _path: Path) raises:
        self.__init__()

    def __init__(out self):
        self.weights = Dict[String, Tensor[DType.float32]]()
        self.weights["attn_q"] = Tensor[DType.float32](TensorShape(2, 2),
             0.3256, -1.8786,
            -0.4062, -0.4507,
        )
        self.weights["attn_k"] = Tensor[DType.float32](TensorShape(2, 2),
             0.6694, -0.7980,
             0.8910,  0.9103,
        )
        self.weights["attn_v"] = Tensor[DType.float32](TensorShape(2, 2),
             0.5933,  1.0371,
            -0.0971,  0.0469,
        )
        self.weights["attn_output"] = Tensor[DType.float32](TensorShape(2, 2),
             0.0713,  0.3269,
             0.0103, -0.0694,
        )
        self.weights["ffn_gate"] = Tensor[DType.float32](TensorShape(2, 2),
             0.5641,  0.4875,
            -1.1172, -1.1583,
        )
        self.weights["ffn_down"] = Tensor[DType.float32](TensorShape(2, 2),
             0.5355, -0.9487,
            -0.6487,  0.1838,
        )
        self.weights["ffn_up"] = Tensor[DType.float32](TensorShape(2, 2),
            -0.6765,  0.7103,
            -0.4643,  0.2860,
        )
        self.weights["attn_norm"] = Tensor[DType.float32](TensorShape(2),
            -0.0766,  0.6322,
        )
        self.weights["ffn_norm"] = Tensor[DType.float32](TensorShape(2),
            -1.0754, -1.1960,
        )
        self.weights["token_embd"] = Tensor[DType.float32](TensorShape(4, 2),
             0.7091, -0.6393,
            -1.0965, -0.0201,
            -0.3484,  0.0024,
            -2.0185, -0.4979,
        )
        self.weights["output_norm"] = Tensor[DType.float32](TensorShape(2),
             1.0476, -0.3264,
        )
        self.weights["output"] = Tensor[DType.float32](TensorShape(4, 2),
             0.1539,  0.0616,
             0.5123, -0.3383,
             0.3272,  0.9645,
            -0.7428, -0.1215,
        )

    fn hyperparams(self) raises -> LlamaHParams:
        return LlamaHParams(
            dims=2,
            n_layers=1,
            n_heads=1,
            vocab_size=4,
            norm_eps=1e-5,
            n_kv_heads=1,
            head_dim=2,
            n_rep=1,
        )

    fn get[
        type: DType
    ](
        mut self, key: String, _layer_idx: Optional[Int] = None
    ) raises -> Tensor[type]:
        constrained[type is DType.float32, "bork"]()
        return self.weights[key].astype[type]()

    def weight(self, graph: Graph, name: String) -> Symbol:
        return graph.constant(self.weights[name])

    def attention(self, graph: Graph) -> Attention:
        params = self.hyperparams()
        return Attention(
            n_heads=params.n_heads,
            n_kv_heads=params.n_kv_heads,
            head_dim=params.head_dim,
            dim=params.dims,
            use_custom_attention=False,
            wq=self.weight(graph, "attn_q").swapaxes(-1, -2),
            wk=self.weight(graph, "attn_k").swapaxes(-1, -2),
            wv=self.weight(graph, "attn_v").swapaxes(-1, -2),
            wo=self.weight(graph, "attn_output").swapaxes(-1, -2),
        )

    def feed_forward(self, graph: Graph) -> FeedForward:
        return FeedForward(
            w1=self.weight(graph, "ffn_gate").swapaxes(-1, -2),
            w2=self.weight(graph, "ffn_down").swapaxes(-1, -2),
            w3=self.weight(graph, "ffn_up").swapaxes(-1, -2),
        )

    def norm(self, graph: Graph, name: String) -> RMSNorm:
        params = self.hyperparams()
        return RMSNorm(params.norm_eps, self.weight(graph, name))

    def transformer(self, graph: Graph) -> Transformer:
        params = self.hyperparams()
        layer = TransformerBlock(
            attention=self.attention(graph),
            feed_forward=self.feed_forward(graph),
            attention_norm=self.norm(graph, "attn_norm"),
            ffn_norm=self.norm(graph, "ffn_norm"),
        )
        return Transformer(
            dim=params.dims,
            n_heads=params.n_heads,
            embedding=Embedding(self.weight(graph, "token_embd")),
            layers=List(layer),
            norm=self.norm(graph, "output_norm"),
            output=self.weight(graph, "output").swapaxes(-1, -2),
            theta=500000.0,
        )


fn test_freqs_cis() raises:
    var g = Graph(
        List[Type]()
    )
    var dummy = Transformer(
        dim=2,
        n_heads=1,
        embedding=Embedding(g.constant(Tensor[DType.float32](10, 10))),
        layers=List[TransformerBlock[DType.float32]](),
        norm=RMSNorm(1e-5, g.constant(Tensor[DType.float32](2))),
        output=g.constant(Tensor[DType.float32](10, 10)),
        theta=500000.0,
    )
    _ = g.output(
        dummy.freqs_cis(g.scalar[DType.int64](0), g.scalar[DType.int64](8), Dim("seq_len"))
    )

    var expected = Tensor[DType.float32](TensorShape(8, 1, 2),
         1.0000,  0.0000,
         0.5403,  0.8415,
        -0.4161,  0.9093,
        -0.9900,  0.1411,
        -0.6536, -0.7568,
         0.2837, -0.9589,
         0.9602, -0.2794,
         0.7539,  0.6570,
    )

    var actual = _testing.execute_nullary(g)

    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-5, rtol=1e-3)


fn test_rope() raises:
    alias bs = 1
    alias seq_len = 2
    alias n_local_kv_heads = 2
    alias head_dim = 2

    var g = Graph(
        List[Type](
            TensorType(DType.float32, bs, seq_len, n_local_kv_heads, head_dim),
            TensorType(DType.float32, 2, 1, 2),
        )
    )
    _ = g.output(rope(x=g[0], freqs_cis=g[1]))

    var x = Tensor[DType.float32](TensorShape(bs, seq_len, n_local_kv_heads, head_dim),
        -1.3140, -1.5004,
         0.4776, -0.2095,

         0.9650,  1.6373,
        -0.0903, -2.1381,
    )
    var freqs_cis = Tensor[DType.float32](TensorShape(2, 1, 2),
         0.4200,  0.9075,
         0.5403,  0.8415,
    )
    var expected = Tensor[DType.float32](TensorShape(bs, seq_len, n_local_kv_heads, head_dim),
         0.8097, -1.8226,
         0.3907,  0.3454,

        -0.8564,  1.6967,
         1.7504, -1.2312,
    )
    var actual = _testing.execute_binary(g, x, freqs_cis)

    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-5, rtol=1e-4)


fn test_rope_batch_size_2() raises:
    alias bs = 2
    alias seq_len = 2
    alias n_local_kv_heads = 1
    alias head_dim = 2

    var g = Graph(
        List[Type](
            TensorType(DType.float32, bs, seq_len, n_local_kv_heads, head_dim),
            TensorType(DType.float32, 2, 1, 2),
        )
    )
    _ = g.output(rope(x=g[0], freqs_cis=g[1]))

    var x = Tensor[DType.float32](TensorShape(bs, seq_len, n_local_kv_heads, head_dim),
         3.4737,  1.4446,
        -2.9905,  0.0881,

         2.1228,  0.5135,
        -2.9574, -1.2562,
    )
    var freqs_cis = Tensor[DType.float32](TensorShape(2, 1, 2),
         1.0000,  0.0000,
         0.5403,  0.8415,
    )
    var expected = Tensor[DType.float32](TensorShape(bs, seq_len, n_local_kv_heads, head_dim),
         3.4737,  1.4446,
        -1.6900, -2.4689,

         2.1228,  0.5135,
        -0.5408, -3.1674,
    )
    var actual = _testing.execute_binary(g, x, freqs_cis)

    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-5, rtol=1e-4)


fn test_rope_seq_len_5() raises:
    alias bs = 1
    alias seq_len = 5
    alias n_local_kv_heads = 1
    alias head_dim = 6

    var g = Graph(
        List[Type](
            TensorType(DType.float32, bs, seq_len, n_local_kv_heads, head_dim),
            TensorType(DType.float32, seq_len, head_dim // 2, 2),
        )
    )
    _ = g.output(rope(x=g[0], freqs_cis=g[1]))

    var freqs_cis = Tensor[DType.float32](TensorShape(seq_len, head_dim // 2, 2),
         1.0000,  0.0000,
         1.0000,  0.0000,
         1.0000,  0.0000,

         0.5403,  0.8415,
         0.8942,  0.4477,
         0.9769,  0.2138,

        -0.4161,  0.9093,
         0.5992,  0.8006,
         0.9086,  0.4177,

        -0.9900,  0.1411,
         0.1774,  0.9841,
         0.7983,  0.6023,

        -0.6536, -0.7568,
        -0.2820,  0.9594,
         0.6511,  0.7590,
    )
    var x = Tensor[DType.float32](TensorShape(bs, seq_len, n_local_kv_heads, head_dim),
         1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345,
        -0.0431, -1.6047, -0.7521,  1.6487, -0.3925, -1.4036,
        -0.7279, -0.5594, -2.3169, -0.2168, -1.3847, -0.8712,
        -0.2234,  1.7174,  0.3189, -0.4245, -0.8286,  0.3309,
        -1.5576,  0.9956, -0.8798, -0.6011, -1.2742,  2.1228,
    )
    var expected = Tensor[DType.float32](TensorShape(bs, seq_len, n_local_kv_heads, head_dim),
         1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345,
         1.3270, -0.9032, -1.4106,  1.1376, -0.0833, -1.4551,
         0.8116, -0.4291, -1.2147, -1.9849, -0.8942, -1.3699,
        -0.0212, -1.7317,  0.4743,  0.2385, -0.8607, -0.2348,
         1.7716,  0.5280,  0.8248, -0.6746, -2.4408,  0.4150,
    )

    var actual = _testing.execute_binary(g, x, freqs_cis)

    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-3)


def test_feed_forward() -> None:
    alias dim = 2
    alias hidden_dim = 2

    llama = NanoLlama()
    g = Graph(TensorType(DType.float32, "batch", "seq_len", dim))
    layer = llama.feed_forward(g)
    g.output(layer(g[0]))

    var input = Tensor[DType.float32](TensorShape(2, 2, dim),
        -1.2620, -2.0678,
        -1.6634,  1.3036,

        -0.0088, -1.1315,
         1.1287,  1.7699,
    )
    var expected = Tensor[DType.float32](TensorShape(2, 2, dim),
         0.1053, -0.1079,
        -0.3632,  0.2142,

         0.4025, -0.1661,
         0.3220, -0.3921,
    )

    var actual = _testing.execute_unary(g, input)

    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-5, rtol=1e-3)


fn test_rms_norm() raises:
    alias dim = 2

    var g = Graph(TensorType(DType.float32, "batch", "seq_len", dim))
    var layer = RMSNorm(
        1e-5,
        g.constant(Tensor[DType.float32](TensorShape(dim), 1.0476, -0.3264)),
    )
    _ = g.output(layer(g[0]))

    var input = Tensor[DType.float32](TensorShape(2, 2, dim),
        -1.2620, -2.0678,
        -1.6634,  1.3036,

        -0.0088, -1.1315,
         1.1287,  1.7699,
    )
    var expected = Tensor[DType.float32](TensorShape(2, 2, dim),
        -0.7718,  0.3940,
        -1.1661, -0.2847,

        -0.0115,  0.4616,
         0.7966, -0.3892,
    )

    var actual = _testing.execute_unary(g, input)

    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_attention_mask() raises:
    var g = Graph(List[Type](
        TensorType(DType.int64, "prev_seq_len", "seq_len"),
        TensorType(DType.bool, "batch", "full_seq_len")
    ))
    var shape = g[0].shape()
    var prev_seq_len = shape[0]
    var seq_len = shape[1]
    _ = g.output(expand_attention_mask(g, g[1], prev_seq_len, seq_len, DType.float32))

    var input = Tensor[DType.int64](TensorShape(0, 2), 0)
    var input_mask = Tensor[DType.bool](TensorShape(1, 2), True)
    var actual = _testing.execute_binary[outtype = DType.float32](g, input, input_mask)

    _testing.assert_tensors_almost_equal(
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
    actual = _testing.execute_binary[outtype = DType.float32](g, input, input_mask)

    _testing.assert_tensors_almost_equal(
        actual,
        Tensor[DType.float32](TensorShape(1, 1, 3), 0, 0, 0),
    )

    # Test the uncommon case with non-zero prev_seq_len and curr_seq_len > 1.
    input = Tensor[DType.int64](TensorShape(1, 2), 0)
    input_mask = Tensor[DType.bool](TensorShape(1, 3), True)
    actual = _testing.execute_binary[outtype = DType.float32](g, input, input_mask)

    _testing.assert_tensors_almost_equal(
        actual,
        Tensor[DType.float32](TensorShape(1, 2, 3), 0, 0, -10000, 0, 0, 0),
    )


def test_attention() -> None:
    alias dim = 2
    alias n_heads = 1
    alias n_kv_heads = 1
    alias head_dim = 2

    # TODO(andrew's fix)
    batch = "batch"
    seq_len = "seq_len"
    start_pos = "start_pos"
    full_seq_len = "full_seq_len"

    g = Graph(
        List[Type](
            TensorType(DType.float32, batch, seq_len, dim),
            TensorType(DType.float32, seq_len, 1, 2),
            TensorType(DType.bool, batch, full_seq_len),
            TensorType(DType.float32, start_pos, 1, batch, n_kv_heads, head_dim),
            TensorType(DType.float32, start_pos, 1, batch, n_kv_heads, head_dim),
        )
    )

    llama = NanoLlama()
    layer = llama.attention(g)
    out = layer(g[0], g[1], g[2], g[3], g[4])
    _ = g.output(List[Symbol](out[0], out[1], out[2]))

    input = Tensor[DType.float32](TensorShape(2, 2, dim),
        -1.2620, -2.0678,
        -1.6634,  1.3036,
        -0.0088, -1.1315,
         1.1287,  1.7699,
    )
    attn_mask = Tensor[DType.bool](TensorShape(2, 2), True)
    # This is a complex tensor of shape (2, 1); we use the last dim as
    # (real, imag).
    # These values generated from the correct freqs_cis given Llama hyperparams.
    freq_cis = Tensor[DType.float32](TensorShape(2, 1, 2),
         1.0000,  0.0000,
         0.5403,  0.8415
    )
    k_cache = Tensor[DType.float32](TensorShape(0, 1, 2, n_kv_heads, head_dim))
    v_cache = Tensor[DType.float32](TensorShape(0, 1, 2, n_kv_heads, head_dim))
    expected_tokens = Tensor[DType.float32](TensorShape(2, 2, dim),
        -0.1979, -0.0316,
        -0.0312, -0.0204,

        -0.1011, -0.0085,
        -0.0874, -0.0067,
    )
    expected_k_cache = Tensor[DType.float32](TensorShape(2, 2, n_kv_heads, head_dim),
         0.8053, -3.0068,
        -0.9151, -1.9720,

         0.8970, -1.0378,
        -2.5569,  0.8611,
    )
    expected_v_cache = Tensor[DType.float32](TensorShape(2, 2, n_kv_heads, head_dim),
        -2.8933,  0.0256,
         0.3651,  0.2227,

        -1.1787, -0.0522,
         2.5052, -0.0266,
    )

    actuals = _testing.execute_n_args(
        g, input, freq_cis, attn_mask, k_cache, v_cache
    )

    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output0"),
        expected_tokens,
        atol=1e-4,
        rtol=1e-4,
    )
    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output1"),
        expected_k_cache,
        atol=1e-4,
        rtol=1e-4,
    )
    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output2"),
        expected_v_cache,
        atol=1e-4,
        rtol=1e-4,
    )


def test_transformer_block() -> None:
    alias dim = 2
    alias n_heads = 1
    alias n_kv_heads = 1
    alias head_dim = 2
    alias hidden_dim = 2

    batch = "batch"
    seq_len = "seq_len"
    prev_seq_len = "prev_seq_len"
    full_seq_len = "full_seq_len"
    cache_type = TensorType(DType.float32, prev_seq_len, 1, batch, n_kv_heads, head_dim)

    g = Graph(
        List[Type](
            TensorType(DType.float32, batch, seq_len, dim),
            TensorType(DType.float32, seq_len, 1, 2),
            TensorType(DType.bool, batch, full_seq_len),
            cache_type,
            cache_type,
        )
    )
    llama = NanoLlama()
    attention = llama.attention(g)
    feed_forward = llama.feed_forward(g)
    attention_norm = RMSNorm(
        1e-5,
        g.constant(Tensor[DType.float32](TensorShape(dim), -0.0766, 0.6322)),
    )
    ffn_norm = RMSNorm(
        1e-5,
        g.constant(Tensor[DType.float32](TensorShape(dim), -1.0754, -1.1960)),
    )
    layer = TransformerBlock(
        attention=attention,
        feed_forward=feed_forward,
        attention_norm=attention_norm,
        ffn_norm=ffn_norm,
    )
    out = layer(g[0], g[1], g[2], g[3], g[4])
    _ = g.output(List[Symbol](out[0], out[1], out[2]))

    input = Tensor[DType.float32](TensorShape(2, 2, dim),
        -1.2620, -2.0678,
        -1.6634,  1.3036,

        -0.0088, -1.1315,
         1.1287,  1.7699,
    )
    var attn_mask = Tensor[DType.bool](TensorShape(2, 2), True)

    # This is a complex tensor of shape (2, 1); we use the last dim as
    # (real, imag).
    # These values generated from the correct freqs_cis given Llama hyperparams.
    freq_cis = Tensor[DType.float32](TensorShape(2, 1, 2),
         1.0000,  0.0000,
         0.5403,  0.8415
    )
    k_cache = Tensor[DType.float32](TensorShape(0, 1, 2, n_kv_heads, head_dim))
    v_cache = Tensor[DType.float32](TensorShape(0, 1, 2, n_kv_heads, head_dim))
    expected_tokens = Tensor[DType.float32](TensorShape(2, 2, dim),
        -1.1102, -2.3339,
        -1.8064,  1.4070,

         0.3818, -1.6129,
         1.2590,  1.6723,
    )
    expected_k_cache = Tensor[DType.float32](TensorShape(2, 2, n_kv_heads, head_dim),
         0.6468, -0.6444,
        -0.6933, -0.0100,

         0.7140, -0.8131,
        -0.8799, -0.1963,
    )
    expected_v_cache = Tensor[DType.float32](TensorShape(2, 2, n_kv_heads, head_dim),
        -0.7580, -0.0413,
         0.6225,  0.0176,

        -0.9267, -0.0420,
         0.7472,  0.0410,
    )

    actuals = _testing.execute_n_args(
        g, input, freq_cis, attn_mask, k_cache, v_cache
    )

    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output0"),
        expected_tokens,
        atol=1e-4,
        rtol=1e-4,
    )
    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output1"),
        expected_k_cache,
        atol=1e-4,
        rtol=1e-4,
    )
    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output2"),
        expected_v_cache,
        atol=1e-4,
        rtol=1e-4,
    )

def test_model() -> None:
    alias dim = 2
    alias layers = 1
    alias n_heads = 1
    alias n_kv_heads = 1
    alias head_dim = 2
    alias hidden_dim = 2

    batch_size = "batch"
    seq_len = "seq_len"
    prev_seq_len = "prev_seq_len"
    full_seq_len = "full_seq_len"

    token_type = TensorType(DType.uint64, batch_size, seq_len)
    amask_type = TensorType(DType.bool, batch_size, full_seq_len)
    cache_type = TensorType(DType.float32, prev_seq_len, layers, batch_size, n_kv_heads, head_dim)
    llama = NanoLlama()
    g = Graph(List[Type](token_type, amask_type, cache_type, cache_type))
    layer = llama.transformer(g)
    outputs = layer(g[0], g[1], g[2], g[3])
    g.output(List(outputs[0], outputs[1], outputs[2]))

    tokens = Tensor[DType.int64](TensorShape(2, 2), 2, 0, 1, 2)
    mask = Tensor[DType.bool](TensorShape(2, 2), True)
    k_cache = Tensor[DType.int64](TensorShape(0, 1, 2, 1, 2))
    v_cache = Tensor[DType.int64](TensorShape(0, 1, 2, 1, 2))

    actuals = _testing.execute_n_args(g, tokens, mask, k_cache, v_cache)

    expected_tokens = Tensor[DType.float32](TensorShape(2, 2, 4),
        -0.2158, -0.6037, -0.6346,  1.0045,
         0.1893,  0.4635,  0.6581, -0.8597,

        -0.2278, -0.6952, -0.5812,  1.0791,
        -0.2158, -0.6040, -0.6345,  1.0047,
    )
    expected_k_cache = Tensor[DType.float32](TensorShape(2, 1, 2, 1, 2),
         0.0676,  0.1021,
         0.0856,  0.0816,

         0.7479,  0.0235,
        -0.0494,  0.1121,
    )
    expected_v_cache = Tensor[DType.float32](TensorShape(2, 1, 2, 1, 2),
         0.0707, -0.0102,
         0.0473, -0.0113,

        -0.6686, -0.0203,
         0.0707, -0.0102,
    )

    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output0"),
        expected_tokens,
        atol=1e-4,
        rtol=1e-4,
    )

    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output1"),
        expected_k_cache,
        atol=1e-4,
        rtol=1e-4,
    )

    _testing.assert_tensors_almost_equal(
        actuals.get[DType.float32]("output2"),
        expected_v_cache,
        atol=1e-4,
        rtol=1e-4,
    )


def main():
    test_freqs_cis()
    test_rope()
    test_rope_batch_size_2()
    test_rope_seq_len_5()
    test_feed_forward()
    test_rms_norm()
    test_attention_mask()
    test_attention()
    test_transformer_block()
    test_model()
