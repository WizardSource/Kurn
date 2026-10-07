"""The hand-run GPU kit: packaging (every file < 1 MB), script syntax, and (KURN_GPU_KIT_DRYRUN=1)
the full dry run from the unzipped kit."""

import os
import shutil
import subprocess
import zipfile

import pytest

from conftest import ROOT

KIT = ROOT / "contrib" / "gpu-check"


def test_release_requires_the_gpu_kit():
    text = (ROOT / "tools" / "make_release.sh").read_text()
    assert "contrib/gpu-check/run_gpu_check.sh" in text and "the hand-run GPU kit must ship" in text


def test_scripts_parse():
    for s in ("run_kit.sh", "run_gpu_check.sh", "run_attn_check.sh", "make_kit.sh"):
        assert subprocess.run(["bash", "-n", str(KIT / s)]).returncode == 0
    for p in ("bench_marlin.py", "bench_attn_torch.py"):
        compile(open(KIT / p).read(), p, "exec")


def _fake_ggml(root):
    for f in ("ggml/CMakeLists.txt", "ggml/include/ggml.h", "ggml/src/ggml.c", "ggml/src/ggml-cuda/fattn.cu", "LICENSE"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(os.urandom(300_000) if f.endswith(".cu") else b"x\n")
    return root


@pytest.mark.skipif(not shutil.which("zip"), reason="zip not installed")
def test_kit_packaging(tmp_path):
    wheels = tmp_path / "wh"
    wheels.mkdir()
    (wheels / "pytest-0-py3-none-any.whl").write_bytes(b"PK")
    env = {**os.environ, "KIT_GGML": str(_fake_ggml(tmp_path / "llama")), "KIT_WHEELS": str(wheels)}
    out = tmp_path / "out"
    subprocess.run(["bash", str(KIT / "make_kit.sh"), str(out)], check=True, capture_output=True, env=env)
    z = out / "kurn-gpu-check.zip"
    for p in out.glob("*.zip"):  # every zip, and every file inside, under 1 MB
        assert p.stat().st_size < 1_000_000, p
        with zipfile.ZipFile(p) as zf:
            assert all(i.file_size < 1_000_000 for i in zf.infolist())
            assert all(n.startswith("kurn-gpu-check/") for n in zf.namelist())
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    for want in ("kurn-gpu-check/run_kit.sh", "kurn-gpu-check/run_gpu_check.sh", "kurn-gpu-check/README.txt",
                 "kurn-gpu-check/bench_marlin.py", "kurn-gpu-check/bench_attn_torch.py",
                 "kurn-gpu-check/kurn/src/kurn/gpu/data/bench_gpu.cu",
                 "kurn-gpu-check/kurn/src/kurn/gpu/data/bench_ggml_attn.cpp", "kurn-gpu-check/kurn/src/kurn/gpu/data/kurn_cuemu.h",
                 "kurn-gpu-check/kurn/examples/gpu/tq2_0_gemv_cuda.kurn", "kurn-gpu-check/kurn/tests/test_gpu_attn.py",
                 "kurn-gpu-check/kurn/tests/conftest.py"):  # fmt: skip
        assert want in names
    assert not any("__pycache__" in n or n.endswith("test_gpu_kit.py") for n in names)
    assert all(
        "/tests/test_gpu_" in n or "/tests/golden/gpu/" in n or n.endswith("conftest.py")
        for n in names
        if "/tests/" in n and not n.endswith("/")
    )
    ggml = sorted(out.glob("kurn-gpu-check-ggml-*.zip"))
    assert len(ggml) >= 2  # 300 KB of incompressible .cu x 1 is split until every part is < 1 MB
    got = set()
    for p in ggml:
        got |= set(zipfile.ZipFile(p).namelist())
    assert {"kurn-gpu-check/ggml-src/CMakeLists.txt", "kurn-gpu-check/ggml-src/COMMIT", "kurn-gpu-check/ggml-src/ggml/include/ggml.h",
            "kurn-gpu-check/ggml-src/ggml/src/ggml-cuda/fattn.cu"} <= got  # fmt: skip
    whl = [n for n in zipfile.ZipFile(out / "kurn-gpu-check-wheels.zip").namelist() if not n.endswith("/")]
    assert whl == ["kurn-gpu-check/wheels/pytest-0-py3-none-any.whl"]


def test_kit_report_has_every_section_and_the_untestable_list(tmp_path):
    import json

    from kurn.gpu import kitreport

    out = tmp_path
    (out / "attn").mkdir()
    (out / "kit.json").write_text(json.dumps({"mode": "quick", "host": "h", "gpu": "NVIDIA A100-SXM4-80GB, 8.0", "kurn": "kurn 0.3"}))
    (out / "arch.json").write_text(json.dumps({"ran_on": "sm_80", "tier": "sm_80", "how": "native SASS", "fatbin": "sm_80 sm_100 sm_120"}))
    (out / "roofline.json").write_text(json.dumps({"hbm_read_gbs": 1800.0}))
    (out / "steps.jsonl").write_text(json.dumps({"step": "GPU pytest subset", "result": "skipped: NO_PYTEST=1", "log": ""}) + "\n")
    cell = {"impl": "kurn", "variant": "default", "nq": 1, "status": "ok", "relerr": 4e-4, "splits": 26,
            "device": "NVIDIA A100-SXM4-80GB", "sm": 80, "native": True, "us_sd": 0.1}  # fmt: skip
    rows = [
        {**cell, "model": "llama3-8b", "kv": kv, "nkv": 16384, "us": us, "GBps": gb}
        for kv, us, gb in (("f16", 50.0, 1400.0), ("q8_0", 30.0, 1200.0))
    ]
    (out / "attn" / "attn_matrix.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = {"impl": "ggml-cuda", "model": "llama3-8b", "nkv": 16384, "nq": 1, "status": "ok", "relerr": 1e-3, "GBps": 1.0}
    base = [{**g, "kv": "f16", "us": 60.0}, {**g, "kv": "q8_0", "us": 45.0}]
    base.append({"impl": "flashinfer", "status": "not installed", "error": "No module named 'torch'"})
    (out / "attn" / "attn_baselines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base))
    abl = {**cell, "model": "llama3-8b", "kv": "q8_0", "nkv": 16384, "variant": "deq 0 (smem pass, run 1)", "us": 66.0}
    (out / "attn" / "attn_ablation.jsonl").write_text(json.dumps(abl) + "\n")
    md = kitreport.write(str(out))
    assert (out / "report.md").read_text() == md
    for want in ("# kurn GPU kit v2 report (QUICK run)", "## Not testable on this box", "FP8 (e4m3) KV attention",
                 "sm_100 (B200 / GB200)", "GPU pytest subset: skipped: NO_PYTEST=1", "| llama3-8b | f16 | 16384 | 50.0", "1.20x", "78%",
                 "flashinfer (No module named", "| llama3-8b | 16384 | 50.0 | 30.0 | 1.67x | 1.33x | 2.00x |",
                 "deq 0 (smem pass, run 1) | 66.0 | +120%", "Q8_0 >= 1.5x kurn F16 (>= 8K): 1/1 cells meet it - MET",
                 "## GEMM / GEMV matmul kit", "## llama.cpp (ggml-cuda) build"):  # fmt: skip
        assert want in md, want


@pytest.mark.skipif(not os.environ.get("KURN_GPU_KIT_DRYRUN"), reason="slow (~2 min): set KURN_GPU_KIT_DRYRUN=1")
def test_kit_dry_run(tmp_path):
    subprocess.run(["bash", str(KIT / "make_kit.sh"), str(tmp_path)], check=True, capture_output=True)
    with zipfile.ZipFile(tmp_path / "kurn-gpu-check.zip") as zf:
        zf.extractall(tmp_path)
    kit = tmp_path / "kurn-gpu-check"
    os.chmod(kit / "run_gpu_check.sh", 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "KURN_CACHE_DIR")}
    r = subprocess.run(["./run_gpu_check.sh"], cwd=kit, env={**env, "DRYRUN": "1", "QUICK": "1", "FORMATS": "q4_0,tq2_0"},
                       capture_output=True, text=True, timeout=1800)  # fmt: skip
    assert r.returncode == 0, r.stdout[-3000:]
    assert ", 0 failures (CPU emulator)" in r.stdout
    tars = list(kit.glob("kurn-gpu-results-*.tar.gz"))
    assert len(tars) == 1
    res = next(p for p in kit.glob("kurn-gpu-results-*") if p.is_dir())
    md = (res / "report.md").read_text()
    assert "No GPU measurements" in md and "unmeasured" in md
