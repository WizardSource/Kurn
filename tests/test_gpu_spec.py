"""`target cuda` specs: validation, legal configurations, covering sets and CLI routing."""

import os
import subprocess
import sys

import pytest

from kurn import hooks
from kurn.cli import main
from kurn.gpu import spec as gs
from kurn.spec import SpecError

from conftest import EXAMPLES, ROOT


def test_defaults_resolve_for_every_format():
    for f, d in gs.FORMATS.items():
        c = gs.resolve({"op": "gemv", "weights": f})
        assert c["target"] == "cuda" and c["arch"] == "sm_80" and c["layout"] == "split"  # 16-byte weight loads
        assert c["xlayout"] == ("split" if d["act"] == "q8_0" else "blocks")
        g = gs.resolve({"op": "gemm", "weights": f})
        assert d["gemm"] and g["bn"] == 8 and g["xin"] == "f32" and g["splitk"] == 0


@pytest.mark.parametrize("bad, msg", [
    ({"op": "gemv", "weights": "q4_0", "tpr": 4, "rpb": 1}, "block size"),
    ({"op": "gemv", "weights": "q4_K", "mins": "bsums", "sub": 4}, "mins=bsums"),
    ({"op": "gemv", "weights": "q4_K", "cols": 8, "unroll": 2}, "cols \\* unroll"),
    ({"op": "gemm", "weights": "q8_0", "bm": 32, "wm": 4}, "multiple of 16"),
    ({"op": "gemm", "weights": "q4_0", "bm": 128, "bn": 128, "wm": 2, "wn": 2, "xin": "f16", "bk": 64}, "registers"),
    ({"op": "gemm", "weights": "q1_0", "bk": 128}, "bk"),
    ({"op": "gemm", "weights": "q4_0", "bn": 128, "bm": 128, "wm": 2, "wn": 4, "xin": "f32"}, "xin=f32"),
    ({"op": "gemm", "weights": "q8_0", "bm": 256, "bn": 256, "wm": 4, "wn": 4, "bk": 256, "stages": 5, "xin": "f16"},
     "shared memory|registers"),
    ({"op": "gemm", "weights": "q4_0", "layout": "split"}, "unknown key"),
    ({"op": "gemv", "weights": "q4_0", "rows": 4}, "unknown key"),
    ({"op": "gemv", "weights": "q4_0", "arch": "sm_70"}, "arch"),
    ({"op": "gemv", "weights": "q4_0", "unpack": "lut"}, "unpack"),
    ({"op": "gemv", "weights": "q5_0"}, "weights"),
    ({"op": "gemv", "weights": "q4_K", "xlayout": "split"}, "xlayout"),
])  # fmt: skip
def test_invalid_specs(bad, msg):
    with pytest.raises(SpecError, match=msg):
        gs.resolve(bad)


def test_legal_and_covering_configs():
    legal = list(gs.legal_configs("gemm", "q4_0"))
    assert len(legal) > 100 and all(gs.gemm_smem(c) <= gs.gemm_smem_limit(c) for c in legal)
    assert any(c["bm"] == 128 and c["bn"] == 128 for c in legal)  # 128x128 CTA tiles (64x32 warp tiles) are legal
    assert not any(c["bm"] // c["wm"] == 64 and c["bn"] // c["wn"] == 64 for c in legal)  # 64x64 warp tiles spill
    for op, f in (("gemv", "q4_K"), ("gemv", "q1_0"), ("gemm", "iq4_nl")):
        cov = gs.covering_configs(op, f, extra=0)
        assert len({gs.config_key(c) for c in cov}) == len(cov)


def test_spec_files_and_routing(capsys):
    assert "cuda" in hooks.TARGET_BACKENDS and "gpu" in hooks.COMMANDS
    gpu = sorted((EXAMPLES / "gpu").glob("*.kurn"))
    assert len(gpu) >= 5
    for p in gpu:
        assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert '"target": "cuda"' in out and "tune space:" in out
    assert main(["gen", str(EXAMPLES / "gpu" / "tq2_0_gemv_cuda.kurn")]) == 0
    assert "kg_gemv" in capsys.readouterr().out
    # target=cuda override on a CPU spec routes to the GPU backend (and its keys then apply)
    assert main(["check", str(EXAMPLES / "q8_0_gemv_vnni16.kurn"), "target=cuda"]) == 2
    assert "unknown key" in capsys.readouterr().err
    # CPU specs are untouched
    assert main(["check", str(EXAMPLES / "q8_0_gemv_vnni16.kurn")]) == 0
    assert '"target": "avx512_vnni"' in capsys.readouterr().out


def test_gpu_cli_misc(capsys):
    assert main(["gpu", "targets"]) == 0
    assert "tq2_0" in capsys.readouterr().out
    assert main(["gpu", "dispatch", "q4_0", "1"]) == 0
    assert '"stock"' in capsys.readouterr().out


def test_gpu_modules_import_without_numpy():
    code = (
        "import sys, importlib, pkgutil\n"
        "sys.modules['numpy'] = None\n"
        "import kurn.gpu, kurn.cli\n"
        "for m in pkgutil.iter_modules(kurn.gpu.__path__):\n"
        "    importlib.import_module('kurn.gpu.' + m.name)\n"
        "from kurn.gpu import codegen, ref, spec\n"
        "codegen.generate(spec.resolve({'op': 'gemv', 'weights': 'e8p'}))\n"
        "ref.problem('e8p', 2, 256, 1)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr
