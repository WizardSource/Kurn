"""`target cuda` specs: validation, legal configurations, covering sets and CLI routing."""

import pytest

from kurn import hooks
from kurn.cli import main
from kurn.gpu import spec as gs
from kurn.spec import SpecError

from conftest import EXAMPLES


def test_defaults_resolve_for_every_format():
    for f, d in gs.FORMATS.items():
        c = gs.resolve({"op": "gemv", "weights": f})
        assert c["target"] == "cuda" and c["arch"] == "sm_80" and c["layout"] == "native"
        if d["gemm"]:
            assert gs.resolve({"op": "gemm", "weights": f})["pipe"] == "reg2"


@pytest.mark.parametrize("bad, msg", [
    ({"op": "gemm", "weights": "q4_K"}, "op gemm supports"),
    ({"op": "gemv", "weights": "q4_0", "tpr": 4, "rpb": 1}, "block size"),
    ({"op": "gemv", "weights": "q4_K", "mins": "bsums", "sub": 4}, "mins=bsums"),
    ({"op": "gemm", "weights": "q8_0", "pipe": "async2", "layout": "native"}, "layout split"),
    ({"op": "gemm", "weights": "q8_0", "bm": 32, "wm": 4}, "multiple of 16"),
    ({"op": "gemm", "weights": "q8_0", "layout": "split", "pipe": "async3", "bm": 128, "bn": 128, "bkb": 4}, "shared memory"),
    ({"op": "gemv", "weights": "q4_0", "rows": 4}, "unknown key"),
    ({"op": "gemv", "weights": "q4_0", "arch": "sm_70"}, "arch"),
    ({"op": "gemv", "weights": "q4_0", "unpack": "lut"}, "unpack"),
    ({"op": "gemv", "weights": "q5_0"}, "weights"),
])  # fmt: skip
def test_invalid_specs(bad, msg):
    with pytest.raises(SpecError, match=msg):
        gs.resolve(bad)


def test_legal_and_covering_configs():
    legal = list(gs.legal_configs("gemm", "q4_0"))
    assert len(legal) > 100 and all(gs.gemm_smem(c) <= gs.SMEM_LIMIT for c in legal)
    for op, f in (("gemv", "q4_K"), ("gemv", "q1_0"), ("gemm", "iq4_nl")):
        cov = gs.covering_configs(op, f, extra=0)
        for k in gs.keys_for(op):  # every legal value of every key appears at least once
            vals = {c[k] for c in cov}
            legal_vals = {c[k] for c in (gs.legal_configs(op, f) if op == "gemm" else cov)}
            assert legal_vals <= vals, (op, f, k)
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
