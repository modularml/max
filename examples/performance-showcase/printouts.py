from itertools import cycle
import random

exclamation_msgs = ["ZAP!", "SHAZAM!", "KAPOW!", "BANG!", "WHAM!"]
framework_labels = dict(tf="TensorFlow", pt="PyTorch", onnx="ONNX")

# TODO: pull these from performance dashboard
expected_speedups = {
    "xX86_64": dict(
        roberta=dict(tf=3.0, pt=1.5),
        clip=dict(tf=2.0, pt=1.5),
    ),
    "xARM_8": dict(
        roberta=dict(tf=2.0, pt=3.5),
        clip=dict(tf=2.5, pt=2.5),
    )
}

def _get_arch():
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        return cpu["arch"]
    except:
        return None 

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
    for framework in filter(lambda x: x in results, ["tf", "pt"]):
        speedup = results["max"] / results[framework]

        modular_txt = f"MAX Engine vs {framework_labels[framework]}:"

        if speedup > 1.5:
            addendum = f"{next(exclamations)} We're {speedup:.2f}x faster!"

        if speedup <= 1.2:
            if _get_arch() in expected_speedups:
                slower.append(
                    f"{expected_speedups[_get_arch()][model][framework]:.2f}x on {framework_labels[framework]}"
                )

            addendum = f"That's about {speedup:.2f}x faster."

        if speedup <= 1:
            addendum = f"Oh, darn that's only {speedup:.2f}x stock performance."

        print(f"{modular_txt} {addendum}")

    if len(slower) > 0:
        slower_txt = " and ".join(slower) + " for " + model
        print()
        print(
            f"Hold on a tick... We normally see speedups of roughly {slower_txt} on {_get_arch()}. Honestly, we would love to hear from you to learn more about the system you're running on! (https://github.com/modularml/max/issues/new/choose)"
        )