"""Workstream `fourbit`: MXFP4 / NVFP4 formats (kurn.mx) and the 4-bit nibble-path
schedule values (unpack=mask16/perm/pair, correction=dpmin; see kurn.generic)."""

from .. import formats, generic, hooks, kernels, mx

formats.FORMATS["mxfp4"] = mx.MXFP4
formats.FORMATS["nvfp4"] = mx.NVFP4

for _w, _short, _doc in (("mxfp4", "mxfp4", "MXFP4 (E2M1 + E8M0 scale per 32) x Q8_0"),
                         ("nvfp4", "nvfp4", "NVFP4 (E2M1 + UE4M3 scale per 16) x Q8_0")):  # fmt: skip
    kernels.KERNELS[("gemv", _w)] = kernels.Kernel(
        "gemv", _w, kernels._GENERIC_TARGETS, kernels._dispatch(None), f"k{_short}_gemv", f"{_short}gemv", _doc
    )
    kernels.KERNELS[("verify", _w)] = kernels.Kernel(
        "verify",
        _w,
        kernels._VERIFY_TARGETS,
        kernels._dispatch(None),
        f"k{_short}_vfy",
        f"{_short}vfy",
        f"{_w} weights x 2-8 activation columns (multi-token verify)",
    )

hooks.EXTRA_INVALID.extend(generic.NIBBLE_INVALID)


def mxfp4_blocks(rng, nblocks, extreme=False):
    """Scales 2^(e-128) for e in [118, 131] (e == 0 is the one inexact corner, see kurn.mx)."""
    out = bytearray()
    for _ in range(nblocks):
        out.append(127 if extreme else 118 + rng.randrange(14))
        out += bytes(0x77 if extreme else rng.randrange(256) for _ in range(16))
    return bytes(out)


def _ue4m3(rng):
    r = rng.random()
    if r < 0.05:
        return rng.choice((0, 0x7F))  # both decode as 0
    if r < 0.15:
        return 1 + rng.randrange(7)  # subnormal
    return 0x28 + rng.randrange(0x30)  # 2^-3 .. 2^2.9


def nvfp4_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        out += bytes(0x7E if extreme else _ue4m3(rng) for _ in range(4))
        out += bytes(0x77 if extreme else rng.randrange(256) for _ in range(32))
    return bytes(out)


hooks.TEST_BLOCKS.update({"mxfp4": mxfp4_blocks, "nvfp4": nvfp4_blocks})

_A5 = {"target": "avx512_vnni", "layout": "i16"}
hooks.GOLDEN.update({
    "q4_0_gemv_avx512_i16_pair_rows2": {"op": "gemv", "weights": "q4_0", **_A5, "unpack": "pair", "rows": 2},
    "q4_K_gemv_avx512_i16_mask16_dpmin_rows1": {"op": "gemv", "weights": "q4_K", **_A5, "unpack": "mask16",
                                                "correction": "dpmin", "rows": 1},
    "iq4_nl_gemv_avx512_i16_perm_rows1": {"op": "gemv", "weights": "iq4_nl", **_A5, "unpack": "perm", "rows": 1},
    "q4_0_gemv_avx2_vnni_i8_mask16_rows2": {"op": "gemv", "weights": "q4_0", "target": "avx2_vnni", "layout": "i8",
                                            "unpack": "mask16", "rows": 2},
    "mxfp4_gemv_scalar": {"op": "gemv", "weights": "mxfp4", "target": "scalar"},
    "mxfp4_gemv_avx512_i16_rows2": {"op": "gemv", "weights": "mxfp4", **_A5, "rows": 2},
    "mxfp4_verify_avx2_vnni_i8_cols4": {"op": "verify", "weights": "mxfp4", "target": "avx2_vnni", "layout": "i8",
                                        "cols": 4},
    "nvfp4_gemv_scalar": {"op": "gemv", "weights": "nvfp4", "target": "scalar"},
    "nvfp4_gemv_avx512_i16_perm_rows1": {"op": "gemv", "weights": "nvfp4", **_A5, "unpack": "perm", "rows": 1},
    "nvfp4_verify_avx512_i16_cols2": {"op": "verify", "weights": "nvfp4", **_A5, "cols": 2},
})  # fmt: skip
