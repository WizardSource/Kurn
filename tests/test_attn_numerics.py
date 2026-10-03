"""Generated attention kernels against the float64 reference in the harness (data/bench_attn.c),
on awkward shapes: query-tile tails, GQA ratios 1-4, causal / explicit fp16 mask / no mask,
decode with automatic and forced KV splits and head blocks, KV-tile tails.

Tolerance (max |out - ref| / max |ref|, reference on the same dequantized K/V): 1e-4 for the
f32 engines, 1e-2 for the bf16 engines (q, k and probabilities rounded to bf16)."""

import pytest

from kurn import attention as A


def _run(cfg):
    c = A.resolve({"op": "attn", **cfg})
    ok, missing = A.runnable(c["target"])
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    row = A.check(A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror")), c)
    assert row["check"] == "ok", row
    assert row["relerr"] < A.TOL[c["target"]]
    return row


@pytest.mark.parametrize("kv", list(A.KV_FORMATS))
@pytest.mark.parametrize("target", list(A.TARGETS))
def test_engine_and_kv_format(target, kv):
    _run({"target": target, "kv": kv})


@pytest.mark.parametrize(
    "cfg",
    [
        {"target": "amx_bf16", "tile_q": 16, "tile_kv": 64},
        {"target": "amx_bf16", "tile_q": 128, "tile_kv": 256, "kv": "q8_0"},
        {"target": "amx_bf16", "split": 4, "dec_rows": 0},
        {"target": "amx_bf16", "split": 1, "dec_rows": 4},
        {"target": "avx512", "tile_q": 256, "tile_kv": 64, "split": 2},
        {"target": "avx512_bf16", "tile_q": 32, "tile_kv": 256, "dec_rows": 0},
        {"target": "amx_bf16", "dk": 64, "kv": "bf16"},
        {"target": "avx512", "dk": 64, "kv": "q8_0", "split": 8},
        {"target": "amx_bf16", "dk": 256, "kv": "q8_0", "tile_kv": 64},
    ],
    ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()),
)
def test_schedule_variants(cfg):
    _run(cfg)


@pytest.mark.parametrize("target", ["amx_bf16", "avx512"])
def test_mla_latent_attention(target):
    """MLA absorbed form: one shared 576-wide latent KV head, v = first 512 values of k."""
    _run({"target": target, "dk": 576, "dv": 512, "mla": 1, "kv": "f16"})
