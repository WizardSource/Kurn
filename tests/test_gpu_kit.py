"""The hand-run GPU kit: packaging (every file < 1 MB), script syntax, and (KURN_GPU_KIT_DRYRUN=1)
the full dry run from the unzipped kit."""

import os
import shutil
import subprocess
import zipfile

import pytest

from conftest import ROOT

KIT = ROOT / "contrib" / "gpu-check"


def test_scripts_parse():
    for s in ("run_gpu_check.sh", "make_kit.sh"):
        assert subprocess.run(["bash", "-n", str(KIT / s)]).returncode == 0
    compile(open(KIT / "bench_marlin.py").read(), "bench_marlin.py", "exec")


@pytest.mark.skipif(not shutil.which("zip"), reason="zip not installed")
def test_kit_packaging(tmp_path):
    subprocess.run(["bash", str(KIT / "make_kit.sh"), str(tmp_path)], check=True, capture_output=True)
    z = tmp_path / "kurn-gpu-check.zip"
    assert z.stat().st_size < 1_000_000
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert all(i.file_size < 1_000_000 for i in zf.infolist())
    for want in ("kurn-gpu-check/run_gpu_check.sh", "kurn-gpu-check/README.txt", "kurn-gpu-check/bench_marlin.py",
                 "kurn-gpu-check/kurn/src/kurn/gpu/data/bench_gpu.cu", "kurn-gpu-check/kurn/src/kurn/gpu/data/kurn_cuemu.h",
                 "kurn-gpu-check/kurn/examples/gpu/tq2_0_gemv_cuda.kurn"):  # fmt: skip
        assert want in names
    assert not any("/tests/" in n or "__pycache__" in n for n in names)


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
