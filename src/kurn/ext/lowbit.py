"""Workstream `lowbit`: 2 bits per weight and below (see kurn.lowbit)."""

from .. import formats, generic, hooks, kernels, lowbit

generic.RECIPES["tq1_0"] = lowbit.TQ1_0_RECIPE
formats.FORMATS.update(lowbit.FORMATS)

kernels.KERNELS[("gemv", "tq1_0")] = kernels.Kernel(
    "gemv", "tq1_0", kernels._GENERIC_TARGETS, kernels._dispatch(None), "ktq10_gemv", "tq10gemv",
    "TQ1_0 (ternary, 1.69 bpw) x Q8_K")  # fmt: skip
kernels.KERNELS[("gemv", "q2_K")] = kernels.Kernel(
    "gemv", "q2_K", ("scalar", "avx512_vnni"), lowbit.lower_q2k_kernel, "kq2k_gemv", "q2kgemv",
    "Q2_K (2-bit k-quant, 2.625 bpw) x Q8_K")  # fmt: skip

hooks.extra_values("layout", lowbit.layouts)
for _key, _legal in lowbit.KEY_LEGAL.items():
    hooks.new_key(_key, _legal, "auto")
hooks.EXTRA_INVALID.extend(lowbit.INVALID)
hooks.RESOLVE_HOOKS.append(lowbit.resolve)
hooks.LOWERINGS["lut"] = lowbit.lower_lut
hooks.LOWERINGS["addsub"] = lowbit.lower_addsub
hooks.TEST_BLOCKS.update(lowbit.TEST_BLOCKS)
hooks.GOLDEN.update(lowbit.GOLDEN)
