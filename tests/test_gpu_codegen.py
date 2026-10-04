"""CUDA code generation: golden files, and real nvcc/ptxas compiles for sm_80, sm_90 and sm_100
(skipped without nvcc). After an intentional codegen change:

    KURN_UPDATE_GOLDEN=1 pytest tests/test_gpu_codegen.py
"""

import os
import subprocess

import pytest

from kurn.gpu import spec as gs
from kurn.gpu.codegen import generate
from kurn.gpu.toolchain import nvcc, ptxas_report, resource_rows

from conftest import ROOT

GOLDEN_DIR = ROOT / "tests" / "golden" / "gpu"
GOLDEN = {
    "q8_0_gemv_native": {"op": "gemv", "weights": "q8_0", "layout": "native", "sub": 2, "unroll": 2, "xlayout": "blocks"},
    "q4_0_gemv_xsplit_cols4": {"op": "gemv", "weights": "q4_0", "xlayout": "split", "cols": 4, "rpb": 2, "unroll": 1},
    "q1_0_gemv_xsplit_sub4": {"op": "gemv", "weights": "q1_0", "xlayout": "split", "sub": 4, "tpr": 64, "rpb": 2},
    "q8_0_gemv_split_sub1_unroll4": {"op": "gemv", "weights": "q8_0", "layout": "split", "sub": 1, "unroll": 4},
    "q4_0_gemv_split_cols4": {"op": "gemv", "weights": "q4_0", "layout": "split", "cols": 4, "unroll": 2},
    "iq4_nl_gemv_native_sub4": {"op": "gemv", "weights": "iq4_nl", "layout": "native", "sub": 4},
    "q4_K_gemv_split_dp4a_tpr64": {"op": "gemv", "weights": "q4_K", "layout": "split", "mins": "dp4a", "tpr": 64, "rpb": 2},
    "q2_0_gemv_native_sub1": {"op": "gemv", "weights": "q2_0", "layout": "native", "sub": 1},
    "tq2_0_gemv_split": {"op": "gemv", "weights": "tq2_0", "layout": "split"},
    "q1_0_gemv_lut_tpr8": {"op": "gemv", "weights": "q1_0", "unpack": "lut", "tpr": 8, "rpb": 16},
    "e8p_gemv_split_sub4": {"op": "gemv", "weights": "e8p", "layout": "split", "sub": 4},
    "q8_0_gemm_bn8": {"op": "gemm", "weights": "q8_0"},
    "q4_0_gemm_128x128_f16": {"op": "gemm", "weights": "q4_0", "bm": 128, "bn": 128, "wm": 2, "wn": 4, "bk": 128, "stages": 3,
                              "xin": "f16"},
    "iq4_nl_gemm_64x32_split4": {"op": "gemm", "weights": "iq4_nl", "bm": 64, "bn": 32, "wm": 2, "wn": 2, "bk": 128,
                                 "splitk": 4, "xin": "f16"},
    "q4_K_gemm_bn16": {"op": "gemm", "weights": "q4_K", "bn": 16, "bm": 64, "wm": 4, "wn": 1},
    "q2_0_gemm_bn8_minb2": {"op": "gemm", "weights": "q2_0", "minb": 2},
    "tq2_0_gemm_bn8": {"op": "gemm", "weights": "tq2_0"},
    "q1_0_gemm_bn8": {"op": "gemm", "weights": "q1_0"},
    "e8p_gemm_64x64_f16": {"op": "gemm", "weights": "e8p", "bm": 64, "bn": 64, "wm": 2, "wn": 2, "bk": 128, "xin": "f16",
                           "stages": 3},
}  # fmt: skip

needs_nvcc = pytest.mark.skipif(not nvcc(), reason="nvcc not installed")


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden(name):
    src = generate(gs.resolve(GOLDEN[name]))
    path = GOLDEN_DIR / f"{name}.cu"
    if os.environ.get("KURN_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(src)
    assert path.exists(), f"missing golden file {path} (run with KURN_UPDATE_GOLDEN=1)"
    assert src == path.read_text(), f"{name}: generated CUDA changed; review and regenerate with KURN_UPDATE_GOLDEN=1"


def test_golden_covers_every_format_and_op():
    have = {(g["op"], g["weights"]) for g in GOLDEN.values()}
    want = {("gemv", f) for f in gs.FORMATS} | {("gemm", f) for f in gs.FORMATS}
    assert want <= have


@needs_nvcc
def test_default_kernels_sass():
    """Default kernels the matrix races: tensor-core MMA, ldmatrix and cp.async in the engine, no local memory anywhere."""
    from kurn.gpu.matrix import default_kernels
    from kurn.gpu.sass import inspect

    for f in ("q4_0", "q1_0", "q4_K", "e8p"):
        for name, (c, _, _) in default_kernels(f, "sm_80").items():
            main = inspect(c)["kg_gemm" if c["op"] == "gemm" else "kg_gemv"]
            k = main["counts"]
            assert k.get("local", 0) == 0, (f, name, k)
            if c["op"] == "gemm":
                assert k.get("mma", 0) and k.get("ldsm", 0) and k.get("cp_async", 0), (f, name, k)
                assert main["loop"]["counts"].get("mma", 0) > 0, (f, name)


@needs_nvcc
@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q1_0"])
def test_xlayout_split_uses_16_byte_activation_loads(fmt):
    """Aligned activations: the default GEMV's hot loop loads activations with 128-bit loads (16-bit loads only for the
    weight scales), and issues far fewer global loads than with ggml q8_0 blocks."""
    from kurn.gpu.sass import inspect

    loops = {
        xl: inspect(gs.resolve({"op": "gemv", "weights": fmt, "xlayout": xl}))["kg_gemv"]["loop"]["counts"] for xl in ("blocks", "split")
    }
    sp, bl = loops["split"], loops["blocks"]
    assert sp.get("ldg128", 0) > bl.get("ldg128", 0) and sp.get("ldg16", 0) <= 4
    assert sp["ldg"] * 2 < bl["ldg"]


@needs_nvcc
@pytest.mark.parametrize("op", ["gemv", "gemm"])
def test_covering_configs_do_not_spill(op):
    """The search space holds no spilling configs: every config of the covering sets compiles without spills or stack."""
    from kurn.gpu.toolchain import build_many

    cs = [c for f, d in gs.FORMATS.items() for c in gs.covering_configs(op, f, extra=4)]
    bad = []
    for c, rep in build_many(cs, lambda c: ptxas_report(c, ("sm_80",))):
        assert not isinstance(rep, Exception), rep
        bad += [(gs.label(c), r) for r in resource_rows(c, rep) if r["spill"]]
    assert not bad, bad[:5]


def test_source_uses_only_sm80_features():
    for g in GOLDEN.values():
        src = generate(gs.resolve(g))
        for banned in ("wgmma", "tcgen05", "cp.async.bulk", "__nv_fp8", "setmaxnreg"):
            assert banned not in src


@needs_nvcc
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_nvcc_compiles_without_spills(name):
    c = gs.resolve(GOLDEN[name])
    rows = resource_rows(c, ptxas_report(c, ("sm_80", "sm_90", "sm_100")))
    assert {r["arch"] for r in rows} == {"sm_80", "sm_90", "sm_100"}
    for r in rows:
        assert r["spill"] == 0, r
        assert r["occupancy"] > 0, r
        assert r["regs"] <= 255


@needs_nvcc
def test_gemm_sass_uses_tensor_cores_and_cp_async(tmp_path):
    from kurn.gpu.toolchain import nvcc_build

    c = gs.resolve({"op": "gemm", "weights": "q4_0", "bm": 128, "bn": 128, "wm": 2, "wn": 4, "xin": "f16", "stages": 3, "bk": 128})
    so, _ = nvcc_build(c, ("sm_80",), out_dir=str(tmp_path))
    dump = subprocess.run([os.path.join(os.path.dirname(nvcc()), "cuobjdump"), "-sass", so], capture_output=True, text=True)
    if dump.returncode:
        pytest.skip("cuobjdump unavailable")
    assert "HMMA.16816.F32" in dump.stdout and "LDSM.16.M88.4" in dump.stdout and "LDGSTS" in dump.stdout
    assert "LDL" not in dump.stdout and "STL" not in dump.stdout


@needs_nvcc
def test_harness_compiles(tmp_path):
    from kurn.gpu.harness import build_harness

    exe = build_harness("sm_80", out_dir=str(tmp_path))
    assert os.access(exe, os.X_OK)
    r = subprocess.run([exe, "info"], capture_output=True, text=True)
    if r.returncode == 4:  # no GPU here: clean JSON error
        assert '"kind": "error"' in r.stdout


@needs_nvcc
@pytest.mark.skipif(not os.environ.get("KURN_LLAMA_DIR"), reason="set KURN_LLAMA_DIR to a llama.cpp built with -DGGML_CUDA=ON")
def test_harness_links_ggml(tmp_path):
    from kurn.gpu.harness import build_harness

    exe = build_harness("sm_80", os.environ["KURN_LLAMA_DIR"], out_dir=str(tmp_path))
    assert "_ggml_" in os.path.basename(exe)
