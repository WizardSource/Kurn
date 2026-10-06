"""Generated CUDA kernels on the CPU warp emulator, against the exact references (no GPU).

Also checks the emulator itself: that it catches the bug classes it is meant to catch (missing
barriers, missing cp.async waits, misaligned vector loads, dropped shuffles, dropped work), and
that its C reference (kurn_gpu_ref.h) and quantizers agree with kurn.formats / ggml's rules.
"""

import pytest

from kurn.gpu import codegen, emu, toolchain
from kurn.gpu import spec as gs

pytestmark = pytest.mark.skipif(bool(toolchain.cxx_problem()), reason=f"CPU emulator unavailable: {toolchain.cxx_problem()}")


def _cases():
    out = []
    for f, d in gs.FORMATS.items():
        out.append({"op": "gemv", "weights": f})
        un = 2 if f in ("q8_0", "e8p") else 1
        out.append({"op": "gemv", "weights": f, "layout": "split", "sub": 2 if f == "q4_K" else 1, "unroll": un, "cols": 4,
                    "rpb": 2})  # fmt: skip
        out.append({"op": "gemv", "weights": f, "layout": "split", "sub": 4, "tpr": 64, "rpb": 2, "cols": 2, "unroll": 2,
                    **({"mins": "dp4a"} if f == "q4_K" else {})})  # fmt: skip
        out.append({"op": "gemv", "weights": f, "layout": "native", "tpr": 8, "rpb": 8, "unroll": 1,
                    **({"unpack": "lut"} if f == "q1_0" else {})})  # fmt: skip
        if d["act"] == "q8_0":  # aligned activation layout (contributed by the user)
            out.append({"op": "gemv", "weights": f, "layout": "split", "sub": 1, "cols": 4, "rpb": 2, "xlayout": "split",
                        "unroll": 2 if f == "q8_0" else 1})  # fmt: skip
            out.append({"op": "gemv", "weights": f, "sub": 4, "tpr": 64, "rpb": 2, "cols": 2, "xlayout": "split", "unroll": 2})
            out.append({"op": "gemv", "weights": f, "layout": "native", "xlayout": "blocks"})
        bk = 256 if f == "q1_0" else 128
        out.append({"op": "gemm", "weights": f})
        out.append({"op": "gemm", "weights": f, "bm": 64, "bn": 32, "wm": 2, "wn": 2, "bk": bk, "stages": 3, "splitk": 2, "xin": "f16"})
        out.append({"op": "gemm", "weights": f, "bm": 32, "bn": 16, "wm": 1, "wn": 2, "bk": 256, "stages": 2, "splitk": 4, "xin": "f16"})
        out.append({"op": "gemm", "weights": f, "bm": 32, "bn": 8, "wm": 2, "wn": 1, "bk": 256, "stages": 2, "splitk": 8, "xin": "f32"})
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
    ({"op": "gemm", "weights": "q4_0", "bm": 64, "bn": 32, "wm": 2, "wn": 2, "xin": "f16", "stages": 3, "splitk": 2},
     "^ (row & 7)) * 16));", ") * 16));", "ldmatrix reads the unswizzled layout"),
    ({"op": "gemm", "weights": "q4_0", "bm": 64, "bn": 32, "wm": 2, "wn": 2, "xin": "f16", "stages": 3, "splitk": 2},
     "KCP_WAIT(STAGES - 2);", "KCP_WAIT(STAGES - 1);", "pipeline reads a stage before its cp.async completes"),
    ({"op": "gemm", "weights": "q4_0"}, "0x64086408u", "0x64076407u", "dequant offset off by one"),
    ({"op": "gemm", "weights": "q4_K"}, "acc[mm][nn][0] += dsc[mm][0] * d[0] - dmn[mm][0] * xs0;",
     "acc[mm][nn][0] += dsc[mm][0] * d[0];", "Q4_K min term dropped"),
    ({"op": "gemm", "weights": "q4_0"}, "    __syncthreads();\n    const int nx", "    const int nx", "missing stage barrier"),
    ({"op": "gemv", "weights": "q8_0", "layout": "split", "sub": 1}, "kld_v4(wq + (size_t)u * 32 + p * 32 + 16)",
     "kld_v4(wq + (size_t)u * 32 + p * 32 + 18)", "misaligned 16-byte load"),
    ({"op": "gemv", "weights": "q4_K", "tpr": 64, "rpb": 2}, "__syncthreads();\n  if (lid == 0", "  if (lid == 0", "cross-warp reduce"),
    ({"op": "gemv", "weights": "q8_0"}, "v += __shfl_xor_sync(KURN_FULL, v, 16);", "", "dropped shuffle"),
    ({"op": "gemv", "weights": "q8_0"}, "for (; w0 < work; w0 += 32) kunit", "for (; w0 < work - 1; w0 += 32) kunit", "dropped tail work"),
])  # fmt: skip
def test_emulator_catches_bugs(ov, old, new, why):
    res = _mutant(gs.resolve(ov), old, new)
    caught = isinstance(res, str) or not all(r["ok"] for r in res)
    assert caught, f"emulator missed: {why}"


def test_cxx_probe_reports_a_broken_compiler(tmp_path, monkeypatch):
    fake = tmp_path / "broken-c++"
    fake.write_text("#!/bin/sh\necho \"fatal error: 'cmath' file not found\" >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("KURN_CXX", str(fake))
    for f in (toolchain.cxx, toolchain.cxx_problem, toolchain.cxx_is_clang):
        f.cache_clear()
    try:
        why = toolchain.cxx_problem()
        assert why and "cmath" in why and "KURN_CXX" in why
        with pytest.raises(toolchain.GpuToolchainError):
            toolchain.emu_build(gs.resolve({"op": "gemv", "weights": "q8_0"}))
    finally:
        for f in (toolchain.cxx, toolchain.cxx_problem, toolchain.cxx_is_clang):
            f.cache_clear()
