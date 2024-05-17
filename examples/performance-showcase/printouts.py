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

import platform
import random
from itertools import cycle

exclamation_msgs = ["ZAP!", "SHAZAM!", "KAPOW!", "BANG!", "WHAM!"]
framework_labels = dict(pt="PyTorch", onnx="ONNX")

# Numbers taken from performance.modular.com
# Caveat: arch is far too coarse a cpu descriptor
#         to predict performance, so this is very rough guidance.
expected_speedups = {
    "X86_64": dict(
        roberta=dict(pt=1.2),
        clip=dict(pt=1.4),
    ),
    "ARM_8": dict(
        roberta=dict(pt=4.0),
        clip=dict(pt=3),
    ),
}


def _get_arch():
    try:
        import cpuinfo

        cpu = cpuinfo.get_cpu_info()
        return cpu["arch"]
    except:
        return None


def _get_platform():
    return platform.system()


def is_mac_silicon():
    # _get_arch() returns None on Apple Silicon, so just check for non-x86 for
    # better stability.
    return _get_arch() != "X86_64" and _get_platform() == "Darwin"


def print_sys_info():
    try:
        import cpuinfo

        cpu = cpuinfo.get_cpu_info()

        keys = ["brand_raw", "arch", "hz_advertised_friendly", "count"]
        labels = {
            "brand_raw": "CPU",
            "arch": "Arch",
            "hz_advertised_friendly": "Clock speed",
            "count": "Cores",
        }

        info_message = "\n" + "-" * 40 + "System Info" + "-" * 40
        info_message += "\n"
        info_message += "\n".join(
            [f'{labels[k]}: {cpu.get(k, "")}' for k in keys]
        )

        print(info_message)
    except:
        pass


def print_speedup_summary(results, model):
    exclamations = cycle(exclamation_msgs)
    [next(exclamations) for i in range(random.randint(0, 4))]

    slower = []
    for framework in filter(lambda x: x in results, ["pt"]):
        speedup = results["max"] / results[framework]

        modular_txt = f"MAX Engine vs {framework_labels[framework]}:"

        if speedup > 1.5:
            addendum = f"{next(exclamations)} MAX is {speedup:.2f}x faster!"
        elif speedup > 1:
            # 1.5 > speedup > 1
            addendum = f"MAX is about {speedup:.2f}x faster."
        else:
            # speedup <= 1.0
            addendum = (
                f"Oh darn, MAX is only {speedup:.2f}x compared to stock"
                " performance."
            )

        curr_arch = _get_arch()
        if speedup <= 1.2 and not is_mac_silicon():
            if curr_arch is not None and curr_arch in expected_speedups:
                slower.append(
                    f"{expected_speedups[curr_arch][model][framework]:.2f}x"
                    f" on {framework_labels[framework]}"
                )

        print(f"{modular_txt} {addendum}")

    if speedup <= 1.2:
        if is_mac_silicon():
            print(
                "We're working hard on improving performance on Apple silicon."
            )
        else:
            slower_txt = " and ".join(slower) + " for " + model
            print()
            print(
                "Hold on a tick... We normally see speedups of roughly"
                f" {slower_txt} on {curr_arch}. Honestly, we would love to hear"
                " from you to learn more about the system you're running on!"
                " (https://github.com/modularml/max/issues/new/choose)"
            )
