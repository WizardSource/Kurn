"""Hadamard-rotated V in the attention harness (bench_attn --v-rot, spec key vrot): the kernel reads the rotated V
cache, the harness rotates the output back, and vqerr (relative RMS error against the unquantized V) shows what the
rotation buys when V has outlier channels (vout) and that it is neutral for iid V."""

import statistics

import pytest

from kurn import attention as A
from kurn.spec import SpecError


def _run(kv, vrot, vout, seeds=(1, 2, 3)):
    c = A.resolve({"op": "attn", "target": "avx512", "kv": kv, "dk": 128, "nq": 16, "nkv": 1024, "vrot": vrot, "vout": vout, "threads": 2})
    ok, missing = A.runnable("avx512")
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    so = A.build(c)
    rows = [A.bench(so, c, "hot", 0, ("--check-toks", "16", "--seed", str(s)), tol=1e-4) for s in seeds]
    assert all(r["check"] == "ok" for r in rows), rows  # kernel vs the (rotated, dequantized) cache, output rotated back
    return statistics.median(r["vqerr"] for r in rows)


@pytest.mark.parametrize("kv", ["q8_0", "k4c_q4"])
def test_rotation_helps_outlier_v_and_is_neutral_otherwise(kv):
    assert _run(kv, 64, 8) < 0.6 * _run(kv, 0, 8)
    plain, rot = _run(kv, 0, 0), _run(kv, 64, 0)
    assert abs(rot - plain) < 0.05 * plain


def test_vrot_spec_validation():
    for bad in ({"vrot": 48}, {"vrot": 256}, {"vrot": 64, "kv": "f16", "dk": 576, "dv": 512, "mla": 1}):
        with pytest.raises(SpecError):
            A.resolve({"op": "attn", "target": "avx512", **{"kv": "q8_0", "dk": 128}, **bad})
    assert A.problem_args(A.resolve({"op": "attn", "target": "avx512", "vrot": 64, "vout": 8}))[-4:] == ["--v-rot", "64", "--v-outlier", "8"]
