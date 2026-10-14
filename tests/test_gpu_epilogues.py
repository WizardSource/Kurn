"""Fused epilogues (kg_epi) and wide verify on the CPU warp emulator, plus their spec rules and ptxas checks.

- epi=swiglu: W = [gate; up] (N = 2F rows), Y[m][f] = silu(gate . x) * (up . x), GEMV and tensor-core GEMM
- epi=fv: Freivalds silent-data-corruption check fused into the GEMM epilogue (no false alarm on clean runs, every block
  checked, a corrupted weight tile caught), with split-K, packed-direct weights and every Freivalds block size
- GEMV cols=16: wide verify (9-16 columns) in one pass over the weights
"""

import pytest

from kurn.gpu import codegen, emu, toolchain
from kurn.gpu import spec as gs
from kurn.gpu.toolchain import nvcc, ptxas_report, resource_rows

needs_emu = pytest.mark.skipif(bool(toolchain.cxx_problem()), reason=f"CPU emulator unavailable: {toolchain.cxx_problem()}")
needs_nvcc = pytest.mark.skipif(not nvcc(), reason="nvcc not installed")

SW = {"op": "gemm", "epi": "swiglu", "wm": 2}
FV = {"op": "gemm", "epi": "fv"}
CASES = [
    *({**SW, "weights": f} for f in ("q8_0", "q4_0", "q4_K", "mxfp4", "nvfp4", "e8p")),
    {**SW, "weights": "tbm", "wload": "reg", "pf": 2},
    {**SW, "weights": "q4_0", "bn": 16, "wload": "reg", "pf": 2, "xin": "f16"},
    {**SW, "weights": "q2_0", "bm": 128, "bn": 32, "wm": 4, "xin": "f16", "bk": 128, "stages": 3},
    {**SW, "weights": "iq4_nl", "bm": 32, "wm": 1, "stages": 2, "xin": "f16"},
    *({"op": "gemv", "epi": "swiglu", "weights": f} for f in ("q8_0", "q4_0", "q4_K", "tq2_0", "e8p", "nvfp4")),
    {"op": "gemv", "epi": "swiglu", "weights": "q1_0", "tpr": 64, "rpb": 2, "cols": 2, "unroll": 1},
    *({**FV, "weights": f} for f in ("q8_0", "q4_0", "q4_K", "q1_0", "mxfp4")),
    {**FV, "weights": "tbm", "wload": "reg", "pf": 2, "fvb": 64},
    {**FV, "weights": "q4_0", "fvb": 64, "splitk": 4},
    {**FV, "weights": "nvfp4", "fvb": 1024, "wload": "reg", "pf": 2, "xin": "f16", "bn": 16},
    {**FV, "weights": "iq4_nl", "bm": 128, "bn": 64, "wm": 4, "wn": 2, "xin": "f16", "bk": 64, "stages": 3, "splitk": 2},
    {**FV, "weights": "tq2_0", "bm": 32, "bn": 16, "wm": 1, "wn": 2, "stages": 2, "splitk": 8, "fvb": 64, "xin": "f16"},
]


@needs_emu
@pytest.mark.parametrize("ov", CASES, ids=lambda o: "-".join(f"{v}" for v in o.values()))
def test_fused_epilogue_matches_reference(ov):
    c = gs.resolve(ov)
    for kw in ({"seed": 1}, {"seed": 2, "extreme": True, "sched": 7}):
        r = emu.run(c, **kw)
        assert r["relerr"] <= emu.TOL and r["quant_ok"], r
        if c.get("epi") == "fv":
            fv = r["fv"]
            assert fv["fails"] == 0 and fv["checks"] == fv["expected"] > 0, fv  # no false alarm, every block checked
            assert fv["detected"] > 0 and fv["worst"] < 0.1, fv  # corrupted tile caught; clean runs far below tau


@needs_emu
@pytest.mark.parametrize("fmt", ["q4_0", "mxfp4", "q1_0"])
def test_wide_verify_single_pass(fmt):
    """cols=16: 9-16 columns in one pass over the weights (grid.y = 1), exact against the reference."""
    from kurn.gpu.ablate import _gemv

    c = _gemv(fmt, "sm_80", cols=16)
    assert c and c["cols"] == 16 and "(M + 16 - 1) / 16" in codegen.generate(c)
    for m in (9, 12, 16):
        assert emu.run(c, m=m, seed=m)["ok"]


def _mutant(c, old, new):
    src = codegen.generate(c)
    assert old in src, old
    orig = toolchain.emu_build
    emu.emu_build = lambda cc, s=None: orig(cc, src.replace(old, new, 1))
    try:
        return [emu.run(c, sched=s) for s in (0, 5)]
    except emu.EmuError as e:
        return str(e)
    finally:
        emu.emu_build = orig


FV256 = {**FV, "weights": "q4_0"}  # fvb 256 > bm 64: several CTAs per block


@needs_emu
@pytest.mark.parametrize("ov, old, new, why", [
    ({**SW, "weights": "q4_0"}, "* acc[mm + 1][nn][e];", "* acc[mm][nn][e];", "swiglu: up product taken from the gate tile"),
    ({**SW, "weights": "q4_0"}, "#define KG_SRT(rt) (((rt) & 1) ? N / 32 + (rt) / 2 : (rt) / 2)", "#define KG_SRT(rt) (rt)",
     "swiglu: repack does not interleave gate and up row tiles"),
    ({**SW, "weights": "tbm", "wload": "reg", "pf": 2}, "if (KG_SRT(1) != 1) prt", "if (0) prt",
     "swiglu tbm: device repack walks records into the wrong packed tiles"),
    ({"op": "gemv", "epi": "swiglu", "weights": "q4_0"}, "(wrow + NR) *", "(wrow + NR - 1) *", "swiglu gemv: up row off by one"),
    (FV256, "atomicAdd(a + 1, (double)((cs & 1u) ? -x0 : x0));", "atomicAdd(a + 1, (double)x0);",
     "fv: X side ignores the secret column signs (false alarms)"),
    ({**FV256, "splitk": 1}, "wi += nct * ", "wi += ", "fv: CTAs of a block overlap their X-side windows"),
    (FV256, "(int)gridDim.z - 1;", "(int)gridDim.z - 2;", "fv: block compared before its last CTA contributed"),
    (FV256, "rv |= 1u << rb;", "rv |= 0u;", "fv: C side drops this lane's rows"),
    ({**FV256, "splitk": 4, "fvb": 64}, "const int w0 = ks0 * (BK / 32)", "const int w0 = 0 * (BK / 32)",
     "fv: split-K X side restarts at k = 0"),
])  # fmt: skip
def test_emulator_catches_epilogue_bugs(ov, old, new, why):
    res = _mutant(gs.resolve(ov), old, new)
    assert isinstance(res, str) or not all(r["ok"] for r in res), f"emulator missed: {why}"


def test_spec_rules():
    for bad, msg in (({**SW, "weights": "q4_0", "wm": 4}, "multiple of 32"), ({**SW, "weights": "q4_0", "splitk": 2}, "no split-K"),
                     ({**FV, "weights": "q4_0", "bm": 128, "wm": 2, "fvb": 64}, "fvb >= bm"),
                     ({"op": "gemv", "weights": "q4_0", "epi": "fv"}, "not allowed"),
                     ({"op": "gemm", "weights": "q4_0", "fvb": 256}, "not allowed"),
                     ({"op": "gemv", "weights": "q1_0", "epi": "swiglu", "sub": 1, "cols": 4, "unroll": 2, "xlayout": "blocks"},
                      "at most 4")):  # fmt: skip
        with pytest.raises(gs.SpecError, match=msg):
            gs.resolve(bad)
    assert gs.resolve({**FV, "weights": "q4_0"})["fvb"] == 256 and gs.resolve({"op": "gemm", "weights": "q4_0"})["fvb"] == 0
    # swiglu GEMVs hold two accumulators and two weight units per column: they get the spill limits of twice the columns
    assert gs.gemv_threads_max(gs.resolve({"op": "gemv", "weights": "q4_0", "epi": "swiglu", "cols": 4, "unroll": 1})) == \
        gs.gemv_threads_max(gs.resolve({"op": "gemv", "weights": "q4_0", "cols": 8, "unroll": 1}))  # fmt: skip
    for op in ("gemv", "gemm"):
        cov = gs.covering_configs(op, "q4_0", extra=0)
        assert {"swiglu"} <= {c["epi"] for c in cov}
    assert {64, 256, 1024} <= {c["fvb"] for c in gs.covering_configs("gemm", "q4_0", extra=0)}


def test_abi_exports_the_epilogue_entry_points():
    for ov, code in (({"op": "gemv", "weights": "q4_0"}, 0), ({"op": "gemv", "weights": "q4_0", "epi": "swiglu"}, 1),
                     ({"op": "gemm", "weights": "q4_0"}, 0), ({**SW, "weights": "q4_0"}, 1), ({**FV, "weights": "q4_0"}, 2)):  # fmt: skip
        src = codegen.generate(gs.resolve(ov))
        assert f"int kg_epi(void) {{ return {code}; }}" in src
        assert "int kg_fv_setup(" in src and "int kg_fv_status(" in src and "int kg_fv_block(void)" in src
    sw = codegen.generate(gs.resolve({**SW, "weights": "q4_0"}))
    assert "N % 32 == 0" in sw and "const int sk = 1;" in sw


@needs_nvcc
def test_epilogue_kernels_do_not_spill_on_three_archs():
    from kurn.gpu.toolchain import build_many

    cs = [gs.resolve(o) for o in CASES] + [c for op in ("gemv", "gemm") for f in ("q4_0", "q4_K", "tbm", "nvfp4")
                                           if f in gs.formats_for(op) for c in gs.covering_configs(op, f, extra=2)
                                           if c["epi"] != "none"]  # fmt: skip
    bad = []
    for c, rep in build_many(cs, lambda c: ptxas_report(c, ("sm_80", "sm_100", "sm_120"))):
        assert not isinstance(rep, Exception), rep
        bad += [(gs.label(c), r) for r in resource_rows(c, rep) if r["spill"]]
    assert not bad, bad[:5]


@needs_nvcc
def test_swiglu_gemv_loads_the_activations_once():
    """The up row's dot products reuse the gate row's activation loads: the hot loop doubles its weight loads only."""
    from kurn.gpu.sass import inspect

    base, sw = (inspect(gs.resolve({"op": "gemv", "weights": "q4_0", "epi": e}))["kg_gemv"]["loop"]["counts"] for e in ("none", "swiglu"))
    assert sw["dp4a"] == 2 * base["dp4a"]
    assert sw["ldg128"] - base["ldg128"] == base["ldg128"] - 8  # 8 activation vectors per iteration, shared
    assert sw["ldg32"] == base["ldg32"]  # activation scales, shared
