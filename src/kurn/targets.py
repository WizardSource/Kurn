"""Target registry: one entry per instruction set kurn can generate for.

Adding a target = one `Target(...)` entry here plus lowerings in codegen.py for
the kernels that support it (listed in kernels.py).
"""

from dataclasses import dataclass

_X86_V3 = frozenset({"avx2", "fma", "f16c", "bmi1", "bmi2", "movbe"})
_X86_V4 = _X86_V3 | {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}


@dataclass(frozen=True)
class Target:
    name: str
    arch: str  # "x86_64" | "aarch64" | "any"
    flags: tuple  # GCC/Clang flags
    requires: frozenset  # /proc/cpuinfo flags needed to run the code
    header: str = ""  # intrinsics header
    doc: str = ""


TARGETS = {
    t.name: t
    for t in (
        # scalar uses the x86-64-v3 baseline on x86 hosts so the auto-vectorizer has AVX2 to work with
        Target("scalar", "any", ("-O3", "-march=x86-64-v3"), _X86_V3, "", "portable C, auto-vectorized"),
        Target("avx2", "x86_64", ("-O3", "-march=x86-64-v3"), _X86_V3, "immintrin.h", "AVX2 + FMA + F16C"),
        Target(
            "avx2_vnni",
            "x86_64",
            ("-O3", "-march=x86-64-v3", "-mavxvnni"),
            _X86_V3 | {"avx_vnni"},
            "immintrin.h",
            "AVX-VNNI (Alder Lake+, Zen 5)",
        ),  # fmt: skip
        Target(
            "avx512_vnni",
            "x86_64",
            ("-O3", "-march=x86-64-v4", "-mavx512vnni"),
            _X86_V4 | {"avx512_vnni"},
            "immintrin.h",
            "AVX-512 + VNNI (Ice Lake+, Zen 4+)",
        ),  # fmt: skip
        Target(
            "amx",
            "x86_64",
            ("-O3", "-march=x86-64-v4", "-mavx512vnni", "-mamx-tile", "-mamx-int8"),
            _X86_V4 | {"avx512_vnni", "amx_tile", "amx_int8"},
            "immintrin.h",
            "AMX-INT8 tiles (Sapphire Rapids+)",
        ),  # fmt: skip
        Target(
            "neon",
            "aarch64",
            ("-O3", "-mcpu=neoverse-n1"),
            frozenset({"asimd", "asimddp", "fphp"}),
            "arm_neon.h",
            "ARMv8.2 NEON + dot product + fp16",
        ),  # fmt: skip
    )
}
