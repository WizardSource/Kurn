"""Generated CUDA kernels on the CPU warp emulator, against the exact references (no GPU).

Also checks the emulator itself: that it catches the bug classes it is meant to catch (missing
barriers, missing cp.async waits, misaligned vector loads, dropped shuffles, dropped work), and
that its C reference (kurn_gpu_ref.h) and quantizers agree with kurn.formats / ggml's rules.
"""

import pytest

from kurn.gpu import codegen, emu, toolchain
from kurn.gpu import spec as gs


def _cases():
    out = []
    for f, d in gs.FORMATS.items():
        out.append({"op": "gemv", "weights": f})
        out.append({"op": "gemv", "weights": f, "layout": "split", "sub": 1, "unroll": 4, "cols": 4, "rpb": 2})
        out.append({"op": "gemv", "weights": f, "layout": "split", "sub": 4, "tpr": 64, "rpb": 2, "cols": 2,
                    **({"mins": "dp4a"} if f == "q4_K" else {})})  # fmt: skip
        out.append({"op": "gemv", "weights": f, "tpr": 8, "rpb": 8, "unroll": 1, **({"unpack": "lut"} if f == "q1_0" else {})})
        if d["gemm"]:
            out.append({"op": "gemm", "weights": f})
            for pipe in ("sync", "async2", "async3"):
                out.append({"op": "gemm", "weights": f, "layout": "split", "pipe": pipe, "bm": 32, "wm": 1, "bn": 16, "wn": 2,
                            "bkb": 4, "pad": 0})  # fmt: skip
    return out


@pytest.mark.parametrize("ov", _cases(), ids=lambda o: "-".join(f"{v}" for v in o.values()))
def test_kernel_matches_reference(ov):
    c = gs.resolve(ov)
    for kw in ({"seed": 1}, {"seed": 2, "extreme": True, "sched": 7}):
        r = emu.run(c, **kw)
        assert r["quant_ok"], "activation quantizer differs from ggml's reference quantizer"
        assert r["relerr"] <= emu.TOL, r


@pytest.mark.parametrize("fmt", sorted(gs.FORMATS))
def test_c_reference_matches_kurn_formats(fmt):
    c = gs.resolve({"op": "gemv", "weights": fmt, "cols": 2})
    a = emu.run(c, n=5, m=3, python_ref=True, seed=4)
    b = emu.run(c, n=5, m=3, seed=4)
    assert a["relerr"] <= emu.TOL and abs(a["relerr"] - b["relerr"]) < 1e-9


def _mutant(c, old, new, scheds=(0, 3, 11)):
    src = codegen.generate(c)
    assert old in src, old
    bad = src.replace(old, new, 1)
    orig = toolchain.emu_build
    emu.emu_build = lambda cc, s=None: orig(cc, bad)
    try:
        return [emu.run(c, sched=s) for s in scheds]
    except emu.EmuError as e:
        return str(e)
    finally:
        emu.emu_build = orig


@pytest.mark.parametrize("ov, old, new, why", [
    ({"op": "gemm", "weights": "q8_0"}, "    __syncthreads();\n    if (ks + 1 < nks)", "    if (ks + 1 < nks)", "read before stored"),
    ({"op": "gemm", "weights": "q8_0"}, "    compute(0);\n    __syncthreads();", "    compute(0);\n", "write after read"),
    ({"op": "gemm", "weights": "q4_0", "layout": "split", "pipe": "async3"}, "KCP_WAIT(1);", "KCP_WAIT(2);", "cp.async not awaited"),
    ({"op": "gemv", "weights": "q8_0", "layout": "split", "sub": 1}, "kld_v4(wq + (size_t)u * 32 + p * 32 + 16)",
     "kld_v4(wq + (size_t)u * 32 + p * 32 + 18)", "misaligned 16-byte load"),
    ({"op": "gemv", "weights": "q4_K", "tpr": 64, "rpb": 2}, "__syncthreads();\n  if (lid == 0", "  if (lid == 0", "cross-warp reduce"),
    ({"op": "gemv", "weights": "q8_0"}, "v += __shfl_xor_sync(KURN_FULL, v, 16);", "", "dropped shuffle"),
    ({"op": "gemv", "weights": "q8_0"}, "if (w < work) kunit", "if (w < work - 1) kunit", "dropped work"),
])  # fmt: skip
def test_emulator_catches_bugs(ov, old, new, why):
    res = _mutant(gs.resolve(ov), old, new)
    caught = isinstance(res, str) or not all(r["ok"] for r in res)
    assert caught, f"emulator missed: {why}"
