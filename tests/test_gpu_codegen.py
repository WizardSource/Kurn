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
    "q8_0_gemv_native": {"op": "gemv", "weights": "q8_0"},
    "q8_0_gemv_split_sub1_unroll4": {"op": "gemv", "weights": "q8_0", "layout": "split", "sub": 1, "unroll": 4},
    "q4_0_gemv_split_cols4": {"op": "gemv", "weights": "q4_0", "layout": "split", "cols": 4},
    "iq4_nl_gemv_native_sub4": {"op": "gemv", "weights": "iq4_nl", "sub": 4},
    "q4_K_gemv_split_dp4a_tpr64": {"op": "gemv", "weights": "q4_K", "layout": "split", "mins": "dp4a", "tpr": 64, "rpb": 2},
    "q2_0_gemv_native_sub1": {"op": "gemv", "weights": "q2_0", "sub": 1},
    "tq2_0_gemv_split": {"op": "gemv", "weights": "tq2_0", "layout": "split"},
    "q1_0_gemv_lut_tpr8": {"op": "gemv", "weights": "q1_0", "unpack": "lut", "tpr": 8, "rpb": 16},
    "e8p_gemv_split_sub4": {"op": "gemv", "weights": "e8p", "layout": "split", "sub": 4},
    "q8_0_gemm_native_reg2": {"op": "gemm", "weights": "q8_0"},
    "q4_0_gemm_split_async3": {"op": "gemm", "weights": "q4_0", "layout": "split", "pipe": "async3", "bn": 64},
    "iq4_nl_gemm_split_sync_minb2": {"op": "gemm", "weights": "iq4_nl", "layout": "split", "pipe": "sync", "minb": 2},
}

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
    want = {("gemv", f) for f in gs.FORMATS} | {("gemm", f) for f, d in gs.FORMATS.items() if d["gemm"]}
    assert want <= have


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

    c = gs.resolve({"op": "gemm", "weights": "q4_0", "layout": "split", "pipe": "async2"})
    so, _ = nvcc_build(c, ("sm_80",), out_dir=str(tmp_path))
    dump = subprocess.run([os.path.join(os.path.dirname(nvcc()), "cuobjdump"), "-sass", so], capture_output=True, text=True)
    if dump.returncode:
        pytest.skip("cuobjdump unavailable")
    assert "IMMA.16832.S8.S8" in dump.stdout and "LDGSTS" in dump.stdout


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
