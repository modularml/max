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
from collections import Dict
from pathlib import Path

from max.engine import InferenceSession, TensorMap
from max.graph import _testing, Graph, TensorType, Symbol, Type
from max.tensor import Tensor, TensorShape
from max._driver import cpu_device

from pipelines.replit.weights.replit_checkpoint import Checkpoint
from pipelines.replit.weights.hyperparams import HyperParams
from pipelines.replit.model.replit import Replit
from pipelines.replit.layers.embedding import SharedEmbedding
from pipelines.replit.layers.attention import GroupedQueryAttention
from pipelines.replit.layers.linear import Linear
from pipelines.replit.layers.block import MPTBlock, MPTMLP
from pipelines.replit.layers.norm import LPLayerNorm


# fmt: off
struct TestCheckpoint(Checkpoint):
    var weights: Dict[String, Tensor[DType.float32]]

    def __init__(inout self, path: Path):
        """Initializes the weights file from a path.

        Args:
            path: Filepath to the model's weights file (unused for testing).
        """
        self.weights = Dict[String, Tensor[DType.float32]]()
        self.weights["transformer.wte.weight"] = Tensor[DType.float32](
            TensorShape(5, 8),
            -0.8603, -1.7125,  0.9310,  1.3544, -0.2425, -0.2951, -0.5038,  1.1447,
            0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
            -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752,
            -2.5224,  2.1028,  0.7568,  0.1173, -1.1854, -0.1747, -0.3490,  0.3125,
            -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822
        )
        self.weights["transformer.blocks.0.norm_1.weight"] = Tensor[
            DType.float32
        ](
            TensorShape(8), 1., 1., 1., 1., 1., 1., 1., 1.
        )
        self.weights["transformer.blocks.0.attn.Wqkv.weight"] = Tensor[
            DType.float32
        ](
            TensorShape(16, 8), -0.0476,  0.1592, -0.0171,  0.0771, -0.2696,  0.2452, -0.0100,  0.0429,
            0.1812,  0.0427, -0.1445,  0.0914, -0.0304, -0.0695,  0.0369,  0.0478,
            -0.0754, -0.0668, -0.3038,  0.1707, -0.1815,  0.2755, -0.3070,  0.1593,
            0.0786,  0.0311,  0.1788,  0.1865, -0.0662, -0.1363,  0.0826, -0.1025,
            0.0856, -0.0047,  0.0288, -0.3141,  0.2436,  0.2577,  0.1278,  0.1883,
            0.2439,  0.3515,  0.2723, -0.0328,  0.2279, -0.2228, -0.2098,  0.2084,
            0.2956,  0.1184,  0.0957, -0.2107, -0.1546,  0.0011, -0.1746, -0.2137,
            -0.3498, -0.1902,  0.2210, -0.2607,  0.0975, -0.3256,  0.1495, -0.0088,
            -0.1649, -0.0496, -0.1562, -0.1919, -0.2841,  0.0966,  0.3396, -0.0129,
            0.0648,  0.0688,  0.2628,  0.2087, -0.1677,  0.2446,  0.0469,  0.1403,
            -0.1690,  0.3326,  0.0010,  0.2642, -0.0774,  0.2465, -0.1054, -0.2960,
            -0.2759, -0.3063,  0.1602, -0.0802,  0.3329,  0.0548, -0.3018,  0.2170,
            0.1203,  0.2776,  0.3162,  0.0234, -0.1581,  0.1047,  0.0586,  0.0887,
            -0.0042,  0.1903,  0.0364,  0.3114, -0.3492,  0.1366, -0.0459,  0.2897,
            0.1986,  0.1852,  0.1488, -0.0134,  0.0679, -0.3049,  0.3002,  0.2837,
            0.2976, -0.3123,  0.2904, -0.2719, -0.2982,  0.1556,  0.2095,  0.0821
        )
        self.weights["transformer.blocks.0.attn.out_proj.weight"] = Tensor[
            DType.float32
        ](
            TensorShape(8, 8), -0.1500,  0.1223, -0.3066,  0.2595,  0.0450, -0.2745, -0.2210, -0.0438,
            -0.2149, -0.0493, -0.2835, -0.1285,  0.1410, -0.1430, -0.3345, -0.2338,
            0.2802, -0.1884,  0.0874,  0.1385,  0.3453, -0.2727, -0.2270, -0.1584,
            -0.0335,  0.0815, -0.1950,  0.3357,  0.2998,  0.1924,  0.0010, -0.1815,
            0.0681, -0.1652,  0.1280,  0.1159,  0.2948, -0.2353, -0.2850,  0.0405,
            -0.0394, -0.0120,  0.2861, -0.1804,  0.0678,  0.0534,  0.1693, -0.3025,
            -0.0262,  0.3524,  0.2530, -0.2841,  0.2843,  0.2771, -0.0985, -0.3345,
            0.2181, -0.1528, -0.3129,  0.1160, -0.1667, -0.2000, -0.1734,  0.1843
        )
        self.weights["transformer.blocks.0.norm_2.weight"] = Tensor[
            DType.float32
        ](
            TensorShape(8), 1., 1., 1., 1., 1., 1., 1., 1.
        )
        self.weights["transformer.blocks.0.ffn.up_proj.weight"] = Tensor[
            DType.float32
        ](
            TensorShape(32, 8), 0.0665,  0.2052,  0.3166, -0.1223, -0.1712,  0.1736,  0.3021, -0.0108,
            0.1479, -0.0357, -0.3083,  0.2699, -0.0323,  0.0449, -0.2675,  0.2078,
            0.1531, -0.0484, -0.2914, -0.0616, -0.1798,  0.3041,  0.2203, -0.2955,
            -0.0599, -0.2264, -0.3238, -0.0196, -0.2894,  0.2592, -0.0912,  0.1247,
            0.2891, -0.1874, -0.3411, -0.1706,  0.0018, -0.0019,  0.1564, -0.1238,
            -0.1550, -0.2406, -0.3502, -0.1223, -0.0740, -0.1992,  0.2419,  0.0942,
            -0.1843, -0.2432,  0.3471, -0.3345,  0.0345, -0.1874,  0.1952,  0.0084,
            0.2487, -0.1231,  0.0311,  0.3100, -0.2304,  0.2620, -0.1075, -0.0337,
            -0.2931, -0.2882, -0.1081,  0.0676,  0.3030, -0.3497, -0.2128, -0.2695,
            0.3367,  0.3481,  0.0595,  0.1308,  0.0587, -0.1473, -0.1568, -0.0218,
            0.0230, -0.2428,  0.1881, -0.0698,  0.2382, -0.1652,  0.0720, -0.1433,
            -0.0824, -0.0229,  0.1497,  0.1837,  0.3522,  0.2404, -0.2588, -0.0095,
            0.3190,  0.0922,  0.2660,  0.2218,  0.0485,  0.0566, -0.0097, -0.1598,
            0.0557,  0.2816, -0.2424,  0.0068,  0.1108,  0.1567,  0.3496, -0.2112,
            0.3152, -0.1581, -0.0658,  0.2063, -0.1005, -0.0218,  0.2004,  0.3293,
            -0.2570,  0.3509,  0.0349,  0.1129,  0.1615,  0.3157, -0.1620, -0.1364,
            0.0040,  0.2997, -0.3406, -0.3042, -0.2524,  0.2658,  0.2445, -0.2563,
            -0.0812,  0.2031,  0.2894,  0.1500, -0.3526,  0.2555, -0.3300, -0.2464,
            0.1849, -0.0597, -0.3493,  0.2930,  0.3115,  0.0697, -0.2846, -0.3229,
            0.3084, -0.2500, -0.1070,  0.0090, -0.2658,  0.1253, -0.0590, -0.1037,
            -0.2675,  0.2378, -0.2095, -0.1856,  0.0962,  0.0990, -0.0010, -0.2059,
            -0.2479,  0.1875,  0.2439,  0.0455, -0.3505, -0.3240,  0.1503, -0.0227,
            0.1173, -0.0974, -0.0481,  0.1781, -0.1429,  0.3253,  0.3032,  0.2056,
            0.2686, -0.0267,  0.3390,  0.0963,  0.0964, -0.1706, -0.2042, -0.1127,
            0.0746,  0.2215, -0.2211, -0.0237,  0.0416,  0.0813,  0.1388,  0.1001,
            0.0953, -0.2312,  0.1480, -0.0646,  0.1702, -0.2270, -0.0252, -0.3279,
            -0.3165,  0.3070,  0.1336, -0.2178,  0.0854, -0.0573,  0.3282, -0.2470,
            -0.1916, -0.3378,  0.2159, -0.1547,  0.0536,  0.1926,  0.0635,  0.2760,
            0.2709,  0.2864,  0.0509, -0.0407,  0.1847, -0.1911,  0.1446, -0.1275,
            -0.1074, -0.0691, -0.1236, -0.2599, -0.1209, -0.0120,  0.1612, -0.1234,
            -0.2399, -0.2877,  0.1236, -0.3511, -0.2893, -0.1304, -0.3163, -0.1136,
            -0.0301,  0.0615,  0.0875, -0.1741,  0.2771,  0.0759,  0.2323,  0.2653
        )
        self.weights["transformer.blocks.0.ffn.down_proj.weight"] = Tensor[
            DType.float32
        ](
            TensorShape(8, 32), -0.1707,  0.0402,  0.1564, -0.0002, -0.1555,  0.1180, -0.0956, -0.0517,
            -0.0780, -0.1161, -0.1454, -0.0437, -0.1452, -0.0330, -0.1260, -0.0558,
            -0.1304, -0.1004, -0.0478,  0.0356,  0.0979,  0.1725,  0.1139, -0.1490,
            -0.0054, -0.0998,  0.0879,  0.0194, -0.0636, -0.0024,  0.1274,  0.1477,
            0.1642,  0.1507, -0.0432, -0.1760, -0.0898, -0.0483,  0.0709, -0.1350,
            0.1161,  0.1623,  0.0866,  0.1533, -0.1465,  0.0213, -0.0016, -0.1756,
            -0.1100, -0.0729, -0.0275, -0.0821, -0.0665,  0.0478, -0.1363, -0.1313,
            0.0028,  0.1071, -0.0276,  0.1760,  0.1053,  0.1251, -0.0733, -0.0929,
            -0.0091, -0.1712, -0.1646,  0.1095, -0.0112,  0.0083,  0.0135,  0.1339,
            -0.0337,  0.0503, -0.0761, -0.0725, -0.0741,  0.0602,  0.0442,  0.0413,
            -0.1285,  0.0855, -0.0937, -0.0881, -0.0734, -0.0014,  0.1717, -0.0283,
            -0.0846,  0.0762, -0.0404,  0.0762,  0.0417,  0.1534, -0.0334, -0.1089,
            0.0575,  0.1343,  0.1048, -0.1338, -0.1341, -0.0705,  0.1333,  0.1749,
            0.0894,  0.1664, -0.0184, -0.1451,  0.0086, -0.0146,  0.1756,  0.0510,
            -0.1359,  0.0163,  0.1579,  0.1316, -0.1290, -0.0379,  0.1392, -0.0103,
            0.0853, -0.0339,  0.1459, -0.1083, -0.0928, -0.0658,  0.0494, -0.1075,
            -0.1394,  0.0886, -0.1220, -0.0076,  0.1119,  0.0519,  0.1052,  0.0411,
            -0.0235,  0.0216,  0.0938,  0.1710,  0.1101,  0.1549,  0.0438, -0.1693,
            -0.1547, -0.1356, -0.1523, -0.0153,  0.1447, -0.0290,  0.0912,  0.0723,
            -0.1636, -0.1584, -0.1752,  0.0400,  0.0865,  0.1223, -0.0530, -0.0178,
            -0.1343, -0.1341,  0.0046,  0.1670,  0.0146,  0.1227, -0.1266, -0.1397,
            0.0453, -0.0207, -0.1236,  0.0552,  0.0436, -0.0810, -0.1609, -0.1649,
            0.0947, -0.0230, -0.0758,  0.1125,  0.0833,  0.1326,  0.0368,  0.1311,
            0.1694,  0.0171,  0.1511, -0.0338, -0.0723, -0.0930,  0.1446, -0.0093,
            0.1124,  0.1725,  0.0166, -0.0554,  0.0088,  0.0748,  0.1589, -0.1204,
            -0.0634,  0.0616,  0.1566, -0.0303,  0.1483,  0.1729,  0.0728,  0.1517,
            0.1572,  0.0907,  0.1045, -0.0946, -0.1151,  0.0521,  0.0480, -0.1754,
            0.0573,  0.0443, -0.1288, -0.0650,  0.0824,  0.1679,  0.0700,  0.1607,
            0.0906, -0.0316, -0.1615, -0.0481, -0.0882,  0.0765, -0.0578, -0.0622,
            -0.0446, -0.1362, -0.1324, -0.1056, -0.1646, -0.1144, -0.1399,  0.0339,
            -0.0576,  0.1284, -0.0444,  0.1576, -0.0531,  0.0660, -0.0335, -0.1493,
            0.0805, -0.0785, -0.0232,  0.1084, -0.1117,  0.0011,  0.0100, -0.0175
        )
        self.weights["transformer.norm_f.weight"] = Tensor[DType.float32](
            TensorShape(8), 1., 1., 1., 1., 1., 1., 1., 1.
        )

    fn __moveinit__(inout self, owned existing: Self):
        self.weights = existing.weights

    fn __copyinit__(inout self, existing: Self):
        self.weights = existing.weights

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
        constrained[type is DType.float32, "Type must be float32."]()
        var found = self.weights.find(key)
        if found:
            return found.value().astype[type]()
        else:
            raise "Could not find key '" + key + "'"


fn get_hyperparams(fused_wkqv: Bool = True) -> HyperParams:
    return HyperParams(
        batch_size=1,
        seq_len=10,
        n_heads=4,
        causal=True,
        alibi=True,
        alibi_bias_max=8,
        num_blocks=1,
        vocab_size=5,
        d_model=8,
        kv_n_heads=2,
        fused_wkqv=fused_wkqv,
    )

fn test_attention() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams(fused_wkqv=True)

    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )

    var attn = GroupedQueryAttention[DType.float32](
        nano_params,
        Linear(g.constant(params.get[DType.float32]("transformer.blocks.0.attn.Wqkv.weight"))),
        Linear(g.constant(params.get[DType.float32]("transformer.blocks.0.attn.out_proj.weight")))
    )
    g.output(attn(g[0])[0])
    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
        -1.8925,  1.7477,  0.6883,  0.1850, -0.8403, -0.0448, -0.1820,  0.3386,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527,
         2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.8331, -1.6811,  0.9493,  1.3706, -0.2184, -0.2707, -0.4784,  1.1619,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527)

    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        -0.0907,  0.3197, -0.0206, -0.1600, -0.0087,  0.1185, -0.0902,  0.0044,
         0.0639,  0.0013,  0.2308,  0.0490,  0.0904,  0.1417,  0.1224, -0.0594,
        -0.0181, -0.0317,  0.3169,  0.0102,  0.1237,  0.2282,  0.1946, -0.0931,
        -0.0245,  0.0890,  0.2498, -0.0354,  0.1130,  0.2140,  0.0853, -0.0684,
         0.0820,  0.1211,  0.3137, -0.0132,  0.1657,  0.2035,  0.0426, -0.0523,
         0.0799,  0.0347,  0.2080,  0.0472,  0.1135,  0.1163,  0.0250, -0.0465,
         0.1290,  0.1221,  0.2316,  0.0458,  0.1065,  0.1555,  0.0463, -0.0507,
         0.0959,  0.0927,  0.2632,  0.0381,  0.1015,  0.1742,  0.0997, -0.0487)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_attention_unfused_matmul() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams(fused_wkqv=False)

    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )

    var attn = GroupedQueryAttention[DType.float32](
        nano_params,
        Linear(g.constant(params.get[DType.float32]("transformer.blocks.0.attn.Wqkv.weight"))),
        Linear(g.constant(params.get[DType.float32]("transformer.blocks.0.attn.out_proj.weight")))
    )
    g.output(attn(g[0])[0])
    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
        -1.8925,  1.7477,  0.6883,  0.1850, -0.8403, -0.0448, -0.1820,  0.3386,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527,
         2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.8331, -1.6811,  0.9493,  1.3706, -0.2184, -0.2707, -0.4784,  1.1619,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527)

    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        -0.0907,  0.3197, -0.0206, -0.1600, -0.0087,  0.1185, -0.0902,  0.0044,
         0.0639,  0.0013,  0.2308,  0.0490,  0.0904,  0.1417,  0.1224, -0.0594,
        -0.0181, -0.0317,  0.3169,  0.0102,  0.1237,  0.2282,  0.1946, -0.0931,
        -0.0245,  0.0890,  0.2498, -0.0354,  0.1130,  0.2140,  0.0853, -0.0684,
         0.0820,  0.1211,  0.3137, -0.0132,  0.1657,  0.2035,  0.0426, -0.0523,
         0.0799,  0.0347,  0.2080,  0.0472,  0.1135,  0.1163,  0.0250, -0.0465,
         0.1290,  0.1221,  0.2316,  0.0458,  0.1065,  0.1555,  0.0463, -0.0507,
         0.0959,  0.0927,  0.2632,  0.0381,  0.1015,  0.1742,  0.0997, -0.0487)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_attention_with_bias() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()

    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8), TensorType(DType.float32, 1, 4, 1, 10)),
    )

    var attn = GroupedQueryAttention[DType.float32](
        nano_params,
        Linear(g.constant[DType.float32](params.get[DType.float32]("transformer.blocks.0.attn.Wqkv.weight"))),
        Linear(g.constant[DType.float32](params.get[DType.float32]("transformer.blocks.0.attn.out_proj.weight")))
    )
    g.output(attn(g[0], g[1])[0])
    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
        -1.8925,  1.7477,  0.6883,  0.1850, -0.8403, -0.0448, -0.1820,  0.3386,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527,
         2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.8331, -1.6811,  0.9493,  1.3706, -0.2184, -0.2707, -0.4784,  1.1619,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527)
    var bias = Tensor[DType.float32](TensorShape(1, 4, 1, 10),
        -2.2500, -2.0000, -1.7500, -1.5000, -1.2500, -1.0000, -0.7500, -0.5000,
        -0.2500,  0.0000, -0.5625, -0.5000, -0.4375, -0.3750, -0.3125, -0.2500,
        -0.1875, -0.1250, -0.0625,  0.0000, -0.1406, -0.1250, -0.1094, -0.0938,
        -0.0781, -0.0625, -0.0469, -0.0312, -0.0156,  0.0000, -0.0352, -0.0312,
        -0.0273, -0.0234, -0.0195, -0.0156, -0.0117, -0.0078, -0.0039,  0.0000)

    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        -0.0907,  0.3197, -0.0206, -0.1600, -0.0087,  0.1185, -0.0902,  0.0044,
         0.0681, -0.0225,  0.2354,  0.0588,  0.0846,  0.1385,  0.1491, -0.0620,
        -0.0258, -0.0646,  0.3360,  0.0141,  0.1253,  0.2275,  0.2158, -0.0887,
        -0.0285,  0.0925,  0.2522, -0.0392,  0.1167,  0.2158,  0.0770, -0.0667,
         0.0927,  0.1336,  0.3022, -0.0113,  0.1621,  0.2030,  0.0433, -0.0587,
         0.1054,  0.0470,  0.1783,  0.0620,  0.0964,  0.1099,  0.0434, -0.0620,
         0.1562,  0.1375,  0.2024,  0.0603,  0.0909,  0.1495,  0.0616, -0.0660,
         0.0996,  0.0751,  0.2700,  0.0461,  0.0993,  0.1702,  0.1142, -0.0468)
    var actual = _testing.execute_binary(g, input, bias)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)



fn test_mpt_block() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var block_prefix = "transformer.blocks.0."

    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )
    var block = MPTBlock[DType.float32].create(params, block_prefix, g, nano_params)
    g.output(block(g[0])[0])
    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
        -2.5224,  2.1028,  0.7568,  0.1173, -1.1854, -0.1747, -0.3490,  0.3125,
        -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752,
         0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
        -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822,
        -0.8603, -1.7125,  0.9310,  1.3544, -0.2425, -0.2951, -0.5038,  1.1447,
        -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822,
        -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752)
    var actual = _testing.execute_unary(g, input)
    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        0.3056, -0.2998, -1.1084, -0.5862,  0.4061, -1.1266, -1.4276, -1.2711,
        -2.2747,  1.9785,  1.0319,  0.1628, -1.5928,  0.1269, -0.0891,  0.5916,
        -0.4814,  1.7713,  1.5678, -0.1263, -0.5337,  0.1245, -0.0338, -0.3837,
         0.2949, -0.5436, -0.8369, -0.4782,  0.5335, -1.0136, -1.2786, -1.3996,
        -0.2665,  0.4962, -0.2078,  0.1107,  0.7217, -0.8450, -1.5606,  0.1508,
        -0.8324, -1.5109,  1.2459,  1.4560,  0.0869, -0.2173, -0.5796,  1.0845,
        -0.2258,  0.5059, -0.2965,  0.2054,  0.6618, -0.9088, -1.5465,  0.1460,
        -0.3838,  1.8936,  1.5217, -0.1029, -0.5370,  0.0715, -0.1384, -0.3549)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)

fn test_mpt_mlp() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )

    var layer = MPTMLP(
        Linear(g.constant[DType.float32](params.get[DType.float32]("transformer.blocks.0.ffn.up_proj.weight"))),
        Linear(g.constant[DType.float32](params.get[DType.float32]("transformer.blocks.0.ffn.down_proj.weight"))))
    g.output(layer(g[0]))

    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        1.9272e+00,  9.3025e-02, -4.7800e-01, -4.1522e-01,  1.2097e+00,
        -6.7925e-01, -1.3663e+00, -2.9106e-01, -1.9211e+00,  1.6819e+00,
         8.1739e-01,  1.6894e-01, -8.4561e-01,  2.3511e-04, -1.2985e-01,
         2.2803e-01, -7.1739e-01,  1.8589e+00,  1.5510e+00, -5.6223e-01,
        -3.9034e-01, -3.1448e-01, -5.6922e-01, -8.5617e-01,  1.9488e+00,
        -3.2597e-01, -1.9494e-01, -3.4731e-01,  1.3098e+00, -6.4897e-01,
        -1.2475e+00, -4.9401e-01,  1.9774e-01,  5.7296e-01,  1.7639e-01,
         1.4493e-01,  1.1745e+00, -9.9278e-01, -2.1208e+00,  8.4709e-01,
        -8.0436e-01, -1.7150e+00,  1.0597e+00,  1.3668e+00, -1.9568e-01,
        -2.3469e-01, -5.0992e-01,  1.0331e+00,  3.1021e-01,  5.9833e-01,
         4.0539e-02,  2.6915e-01,  1.0911e+00, -1.0690e+00, -2.0982e+00,
         8.5784e-01, -5.6874e-01,  1.9744e+00,  1.4302e+00, -5.2662e-01,
        -4.2639e-01, -3.8885e-01, -6.9075e-01, -8.0325e-01)
    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        -0.5217,  0.1831, -0.2177,  0.2597,  0.0914, -0.0921,  0.0977, -0.5139,
         0.1861, -0.1233,  0.0451, -0.0026, -0.4976,  0.1611,  0.1377,  0.3399,
        -0.1385, -0.0538,  0.0466,  0.0997, -0.4558,  0.1373,  0.2126,  0.0868,
        -0.5973,  0.1713, -0.2165,  0.2418,  0.0968, -0.0738,  0.0705, -0.5685,
        -0.2498,  0.2672, -0.1975,  0.1602,  0.0758, -0.0496, -0.0103, -0.2805,
        -0.0517,  0.1596,  0.1088,  0.0626,  0.2157, -0.0400, -0.1024, -0.0153,
        -0.2601,  0.2782, -0.2054,  0.2049,  0.0747, -0.0693,  0.0034, -0.2905,
        -0.1572, -0.0580,  0.0547,  0.0938, -0.4369,  0.1387,  0.2027,  0.0693)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_shared_embedding() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var g = Graph(
        List[Type](TensorType(DType.int32, 1, 8)),
    )
    var w = g.constant[DType.float32](params.get[DType.float32]("transformer.wte.weight"))
    var layer = SharedEmbedding(w)
    g.output(layer(g[0]))
    var input = Tensor[DType.int32](TensorShape(1, 8),
        1, 3, 2, 1, 4, 0, 4, 2)
    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
        -2.5224,  2.1028,  0.7568,  0.1173, -1.1854, -0.1747, -0.3490,  0.3125,
        -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752,
         0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
        -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822,
        -0.8603, -1.7125,  0.9310,  1.3544, -0.2425, -0.2951, -0.5038,  1.1447,
        -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822,
        -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)

fn test_shared_embedding_unembed() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )
    var w = g.constant[DType.float32](params.get[DType.float32]("transformer.wte.weight"))
    var layer = SharedEmbedding(w)
    g.output(layer(g[0], True))
    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        1.4109,  0.5062, -0.7023,  0.0782,  1.5611, -0.7295, -1.1792, -0.9454,
        -1.7746,  1.5395,  0.8181,  0.1406, -1.2493,  0.1024, -0.0443,  0.4675,
        -0.8558,  1.7665,  1.5831, -0.4237, -0.9087, -0.1369, -0.2978, -0.7267,
         1.3464,  0.0786, -0.3724,  0.1633,  1.7200, -0.6418, -1.0637, -1.2304,
        -0.1199,  0.9884, -0.0648,  0.4224,  1.2894, -0.9741, -2.0008,  0.4595,
        -0.8964, -1.5943,  1.1241,  1.3843, -0.0221, -0.3162, -0.6531,  0.9737,
        -0.0332,  1.0200, -0.2109,  0.6005,  1.1947, -1.0703, -1.9450,  0.4442,
        -0.7281,  1.8884,  1.4836, -0.3958, -0.9121, -0.2083, -0.4312, -0.6964)
    var expected = Tensor[DType.float32](TensorShape(1, 8, 5),
        -3.2800,  5.2049,  0.3586, -4.6237,  3.0359,  0.6726, -4.4877,  4.4617,
         9.9737, -0.3301, -1.8099, -2.4457,  6.1897,  7.9996, -0.3979, -2.5187,
         5.1450, -0.0361, -5.4336,  2.5538,  0.4311,  2.9250,  2.3859,  1.8649,
         5.1234,  7.9651, -0.9179, -1.6446,  0.5353,  1.3245,  0.4133,  2.9960,
         2.2226,  1.7278,  5.1146, -2.0596, -2.1093,  6.3143,  7.9343, -0.0695)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_norm() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )
    var w = g.constant[DType.float32](params.get[DType.float32]("transformer.norm_f.weight"))
    var layer = LPLayerNorm[DType.float32](w, nano_params)
    g.output(layer(g[0]))

    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
        -2.5224,  2.1028,  0.7568,  0.1173, -1.1854, -0.1747, -0.3490,  0.3125,
        -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752,
         0.9180, -0.8026, -0.8702, -0.6859,  0.3235, -1.1530, -1.4351, -0.7616,
        -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822,
        -0.8603, -1.7125,  0.9310,  1.3544, -0.2425, -0.2951, -0.5038,  1.1447,
        -0.1004,  0.1079, -0.3241, -0.0315,  0.4788, -1.0013, -1.5907,  0.4822,
        -0.3225,  1.8615,  1.2045, -0.2332, -0.2013, -0.2403, -0.4407, -0.3752)
    var expected = Tensor[DType.float32](TensorShape(1, 8, 8),
        2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
        -1.8925,  1.7477,  0.6883,  0.1850, -0.8403, -0.0448, -0.1820,  0.3386,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527,
         2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.8331, -1.6811,  0.9493,  1.3706, -0.2184, -0.2707, -0.4784,  1.1619,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)


fn test_linear() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var g = Graph(
        List[Type](TensorType(DType.float32, 1, 8, 8)),
    )
    var w = g.constant[DType.float32](params.get[DType.float32]("transformer.blocks.0.attn.Wqkv.weight"))
    var layer = Linear(w)
    g.output(layer(g[0]))

    var input = Tensor[DType.float32](TensorShape(1, 8, 8),
        2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
        -1.8925,  1.7477,  0.6883,  0.1850, -0.8403, -0.0448, -0.1820,  0.3386,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527,
         2.0151, -0.3334, -0.4256, -0.1741,  1.2037, -0.8116, -1.1967, -0.2774,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.8331, -1.6811,  0.9493,  1.3706, -0.2184, -0.2707, -0.4784,  1.1619,
         0.2181,  0.5272, -0.1138,  0.3204,  1.0776, -1.1188, -1.9934,  1.0827,
        -0.5880,  2.0923,  1.2860, -0.4784, -0.4392, -0.4871, -0.7330, -0.6527)
    var expected = Tensor[DType.float32](TensorShape(1, 8, 16),
        -6.7860e-01,  3.5888e-01, -1.4896e-01, -2.6583e-05,  9.5386e-02,
         9.1252e-01,  6.3338e-01, -4.8499e-01, -1.0391e+00, -5.3596e-01,
        -5.8285e-01,  1.4914e-01, -3.5880e-01, -6.9825e-01,  1.6870e-01,
        -1.3115e-01,  6.0272e-01, -3.1272e-01,  9.8407e-02,  7.5168e-02,
        -3.8424e-01,  2.6132e-01, -2.3633e-01,  3.3594e-01,  2.5060e-01,
         3.8603e-01,  9.2364e-01, -7.1538e-02,  6.2699e-01,  8.1696e-01,
         4.5791e-02, -7.2616e-01,  2.8051e-01, -2.5780e-01, -5.0121e-01,
         2.6139e-01, -3.2196e-01,  9.8409e-01,  6.3261e-01,  2.2859e-01,
        -2.7867e-01,  1.7252e-01,  8.5455e-01, -3.2758e-01,  8.2311e-01,
         2.2986e-01,  1.8196e-01, -4.7686e-01, -6.7860e-01,  3.5888e-01,
        -1.4896e-01, -2.6583e-05,  9.5386e-02,  9.1252e-01,  6.3338e-01,
        -4.8499e-01, -1.0391e+00, -5.3596e-01, -5.8285e-01,  1.4914e-01,
        -3.5880e-01, -6.9825e-01,  1.6870e-01, -1.3115e-01, -3.9827e-01,
         1.3095e-01,  3.1824e-01, -1.2153e-01, -1.6442e-01,  1.3357e+00,
        -2.6611e-03, -1.2344e-01, -1.2110e+00, -3.0859e-01, -2.4654e-01,
         8.6839e-01, -1.6418e-01,  7.1070e-02,  2.4276e-01, -1.0441e+00,
        -9.1401e-02, -1.7130e-01,  4.1770e-01,  2.0033e-01, -4.3189e-01,
        -2.2751e-01, -7.7455e-01,  4.4874e-01, -3.3209e-01,  4.7687e-01,
        -3.9860e-01,  1.0959e+00, -1.5344e-01,  5.4279e-01, -1.0018e-01,
         1.9826e-01, -3.9827e-01,  1.3095e-01,  3.1824e-01, -1.2153e-01,
        -1.6442e-01,  1.3357e+00, -2.6611e-03, -1.2344e-01, -1.2110e+00,
        -3.0859e-01, -2.4654e-01,  8.6839e-01, -1.6418e-01,  7.1070e-02,
         2.4276e-01, -1.0441e+00,  2.8051e-01, -2.5780e-01, -5.0121e-01,
         2.6139e-01, -3.2196e-01,  9.8409e-01,  6.3261e-01,  2.2859e-01,
        -2.7867e-01,  1.7252e-01,  8.5454e-01, -3.2758e-01,  8.2311e-01,
         2.2986e-01,  1.8196e-01, -4.7686e-01)
    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(actual, expected, atol=1e-4, rtol=1e-4)

fn test_replit_logits() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var replit = Replit[TestCheckpoint, DType.float32](nano_params)
    var g = replit.build_graph(
        "Nano Replit",
        params,
        with_attention_mask=True,
        use_cache=True
    )

    # Test with empty kv cache.
    var input = Tensor[DType.int64](TensorShape(1, 7), 1, 3, 2, 1, 4, 0, 2)
    var attention_mask = Tensor[DType.bool](TensorShape(1, 7), True)
    var kv_cache = replit.create_empty_cache(cpu_device())
    var k_cache = Tensor[DType.float32](kv_cache[0].spec())
    var v_cache = Tensor[DType.float32](kv_cache[1].spec())
    var result_map = execute_replit(g, input, attention_mask, k_cache, v_cache)

    var expected_logits = Tensor[DType.float32](TensorShape(1, 5),
        -1.9980, -2.2461,  6.2784,  8.0301, -0.1669)
    var logits = result_map.get[DType.float32]("output0")
    _testing.assert_tensors_almost_equal(logits, expected_logits, atol=1e-4, rtol=1e-4)

    var expected_k_cache = Tensor[DType.float32](TensorShape(1, 1, 2, 2, 7),
        -1.0390,  0.2506, -0.2787, -1.0390, -1.2110, -0.3321, -0.2787, -0.5360,
         0.3860,  0.1726, -0.5360, -0.3086,  0.4769,  0.1726, -0.5828,  0.9236,
         0.8546, -0.5828, -0.2465, -0.3986,  0.8546,  0.1491, -0.0715, -0.3276,
         0.1491,  0.8684,  1.0959, -0.3276)
    var new_k_cache = result_map.get[DType.float32]("output1")
    _testing.assert_tensors_almost_equal(new_k_cache, expected_k_cache, atol=1e-4, rtol=1e-4)

    var expected_v_cache = Tensor[DType.float32](TensorShape(1, 1, 2, 7, 2),
        -0.3588, -0.6982,  0.6270,  0.8170,  0.8232,  0.2299, -0.3588, -0.6982,
        -0.1642,  0.0711, -0.1535,  0.5428,  0.8232,  0.2299,  0.1687, -0.1312,
         0.0458, -0.7262,  0.1820, -0.4769,  0.1687, -0.1312,  0.2427, -1.0441,
        -0.1002,  0.1982,  0.1820, -0.4769)
    var new_v_cache = result_map.get[DType.float32]("output2")
    _testing.assert_tensors_almost_equal(new_v_cache, expected_v_cache, atol=1e-4, rtol=1e-4)


fn test_replit_logits_with_prev_cache() raises:
    var params = TestCheckpoint(Path(""))
    var nano_params = get_hyperparams()
    var replit = Replit[TestCheckpoint, DType.float32](nano_params)
    var g = replit.build_graph(
        "Nano Replit",
        params,
        with_attention_mask=True,
        use_cache=True
    )

    # Test with new inputs and the kv cache computed previously.
    var input2 = Tensor[DType.int64](TensorShape(1, 2), 2, 3)
    var k_cache = Tensor[DType.float32](TensorShape(1, 1, 2, 2, 7),
        -1.0390,  0.2506, -0.2787, -1.0390, -1.2110, -0.3321, -0.2787, -0.5360,
         0.3860,  0.1726, -0.5360, -0.3086,  0.4769,  0.1726, -0.5828,  0.9236,
         0.8546, -0.5828, -0.2465, -0.3986,  0.8546,  0.1491, -0.0715, -0.3276,
         0.1491,  0.8684,  1.0959, -0.3276)
    var v_cache = Tensor[DType.float32](TensorShape(1, 1, 2, 7, 2),
        -0.3588, -0.6982,  0.6270,  0.8170,  0.8232,  0.2299, -0.3588, -0.6982,
        -0.1642,  0.0711, -0.1535,  0.5428,  0.8232,  0.2299,  0.1687, -0.1312,
         0.0458, -0.7262,  0.1820, -0.4769,  0.1687, -0.1312,  0.2427, -1.0441,
        -0.1002,  0.1982,  0.1820, -0.4769)
    var attention_mask2 = Tensor[DType.bool](TensorShape(1, 9), True)
    var result_map2 = execute_replit(g, input2, attention_mask2, k_cache, v_cache)
    var expected_logits2 = Tensor[DType.float32](TensorShape(1, 5),
        0.8508, -4.5605,  4.4140,  9.9478, -0.3976)
    var logits2 = result_map2.get[DType.float32]("output0")
    _testing.assert_tensors_almost_equal(logits2, expected_logits2, atol=1e-4, rtol=1e-4)
    var expected_k_cache = Tensor[DType.float32](TensorShape(1, 1, 2, 2, 9),
        -1.0390,  0.2506, -0.2787, -1.0390, -1.2110, -0.3321, -0.2787, -0.2787,
         0.2506, -0.5360,  0.3860,  0.1726, -0.5360, -0.3086,  0.4769,  0.1726,
         0.1726,  0.3860, -0.5828,  0.9236,  0.8546, -0.5828, -0.2465, -0.3986,
         0.8546,  0.8546,  0.9236,  0.1491, -0.0715, -0.3276,  0.1491,  0.8684,
         1.0959, -0.3276, -0.3276, -0.0715)
    var new_k_cache = result_map2.get[DType.float32]("output1")
    _testing.assert_tensors_almost_equal(new_k_cache, expected_k_cache, atol=1e-4, rtol=1e-4)

    var expected_v_cache = Tensor[DType.float32](TensorShape(1, 1, 2, 9, 2),
        -0.3588, -0.6982,  0.6270,  0.8170,  0.8232,  0.2299, -0.3588, -0.6982,
        -0.1642,  0.0711, -0.1535,  0.5428,  0.8232,  0.2299,  0.8232,  0.2299,
         0.6270,  0.8170,  0.1687, -0.1312,  0.0458, -0.7262,  0.1820, -0.4769,
         0.1687, -0.1312,  0.2427, -1.0441, -0.1002,  0.1982,  0.1820, -0.4769,
         0.1820, -0.4769,  0.0458, -0.7262)
    var new_v_cache = result_map2.get[DType.float32]("output2")
    _testing.assert_tensors_almost_equal(new_v_cache, expected_v_cache, atol=1e-4, rtol=1e-4)

# fmt: on


fn execute_replit(
    g: Graph,
    input: Tensor[DType.int64],
    attention_mask: Tensor[DType.bool],
    k_cache: Tensor[DType.float32],
    v_cache: Tensor[DType.float32],
) raises -> TensorMap:
    var session = InferenceSession()
    var model = session.load(g)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", input)
    input_map.borrow("input1", attention_mask)
    input_map.borrow("input2", k_cache)
    input_map.borrow("input3", v_cache)

    var result_map = model.execute(input_map)
    return result_map^


fn main() raises:
    test_attention()
    test_attention_unfused_matmul()
    test_attention_with_bias()
    test_mpt_block()
    test_mpt_mlp()
    test_shared_embedding()
    test_shared_embedding_unembed()
    test_norm()
    test_linear()
    test_replit_logits()
    test_replit_logits_with_prev_cache()
