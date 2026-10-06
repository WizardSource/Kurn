"""Pre-RoPE 4-bit per-channel K formats (k4c_q4, k4c_q8): RoPE variants, group / tail edges,
spec validation, and deliberately broken kernels the harness must reject."""

import pytest

from kurn import attention as A
from kurn.spec import SpecError


def _check(cfg):
    c = A.resolve({"op": "attn", **cfg})
    ok, missing = A.runnable(c["target"])
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    row = A.check(A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror")), c)
    assert row["check"] == "ok", row
    return row


@pytest.mark.parametrize(
    "cfg",
    [
        {"target": "avx512", "kv": "k4c_q4", "rope_mode": 1},
        {"target": "amx_bf16", "kv": "k4c_q8", "rope_mode": 1},
        {"target": "avx512", "kv": "k4c_q4", "dk": 64, "rope_dim": 32},
        {"target": "amx_bf16", "kv": "k4c_q4", "dk": 256, "tile_kv": 64},
        {"target": "avx512", "kv": "k4c_q8", "split": 8, "rope_base": 10000},
        {"target": "avx512_bf16", "kv": "k4c_q4", "dec_rows": 0, "tile_kv": 256},
    ],
    ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()),
)
def test_variants(cfg):
    _check(cfg)


def test_spec_rejects_mla_and_bad_rope():
    with pytest.raises(SpecError, match="MLA"):
        A.resolve({"op": "attn", "target": "avx512", "kv": "k4c_q4", "dk": 576})
    with pytest.raises(SpecError, match="rope_dim"):
        A.resolve({"op": "attn", "target": "avx512", "kv": "k4c_q4", "rope_dim": 48})


@pytest.mark.parametrize(
    "old,new",
    [
        ("if (rd > 0) ka_rope_row(", "if (0) ka_rope_row("),
        ("_mm512_srli_epi32(x, 4)), sc[c / 16 + 1]", "_mm512_and_si512(x, _mm512_set1_epi32(15))), sc[c / 16 + 1]"),
        ("if (j < nfull) {", "if (j < nfull + 32) {"),
    ],
    ids=["no-rope", "nibble-order", "tail-read-as-block"],
)
def test_harness_rejects_broken_kernels(old, new):
    c = A.resolve({"op": "attn", "target": "avx512", "kv": "k4c_q4"})
    ok, missing = A.runnable(c["target"])
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    src = A.generate(c)
    assert old in src
    so = A._compile(src.replace(old, new), A.TARGETS["avx512"][0], "attn_k4c_broken")
    assert A.check(so, c)["check"] == "FAIL"
