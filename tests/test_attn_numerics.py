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


@pytest.mark.parametrize("kv", ["f16", "q8_0"])
@pytest.mark.parametrize("target", ["amx_bf16", "avx512_bf16"])
def test_k_offset_centering(target, kv):
    """Constant K channel offsets (Qwen2-style key bias, harness --k-bias) must not add bf16 rounding
    error: kcenter subtracts them before rounding. Without it, relerr is ~0.1-0.4 at bias 64-316."""
    c = A.resolve({"op": "attn", "target": target, "kv": kv})
    ok, missing = A.runnable(target)
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    so = A.build(c)
    # the second shape runs the tile engine with 3 KV splits, so the LSE merge sees the q.mu correction
    for sh in ({"nq": 100, "nkv": 700, "heads": 8, "kv_heads": 2}, {"nq": 1, "nkv": 2000, "heads": 16, "kv_heads": 1}):
        cc = {**c, **A.PROBLEM, "pos0": -1, **sh, "threads": 3}
        row = A.bench(so, cc, "hot", 0, ("--check-toks", "16", "--k-bias", "100"))
        assert row["check"] == "ok", row


@pytest.mark.parametrize(
    "cfg",
    [
        {"target": "amx_bf16", "kv": "f16"},
        {"target": "amx_bf16", "kv": "q8_0", "tile_q": 128, "tile_kv": 256},
        {"target": "amx_bf16", "kv": "bf16", "dk": 64, "tile_q": 16, "tile_kv": 64},
        {"target": "avx512_bf16", "kv": "f16"},
        {"target": "amx_bf16", "kv": "f16", "dk": 576, "dv": 512, "mla": 1},
    ],
    ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()),
)
def test_packed_kv(cfg):
    """kattn_pack + kattn_packed (AMX-ready K/V written at cache-write time) on the awkward shapes,
    packed in one go and as two appends (the second recomputes the K mean), with and without a K offset."""
    mla = cfg.pop("mla", 0)
    c = A.resolve({"op": "attn", **cfg})
    ok, missing = A.runnable(c["target"])
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    so = A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror"))
    for sh in A.CHECK_SHAPES:
        cc = {**c, **A.PROBLEM, "pos0": -1, **sh, "threads": 3}
        if mla:
            cc.update(mla=1, kv_heads=1)
        for extra in ((), ("--packed-split", "33"), ("--k-bias", "100", "--packed-split", "7")):
            row = A.bench(so, cc, "hot", 0, ("--check-toks", "40", "--packed", *extra))
            assert row["check"] == "ok", (sh, extra, row["relerr"])


@pytest.mark.parametrize("kv", ["f16", "q8_0"])
def test_qk_int8(kv):
    """qk int8 (Q K^T on AMX-INT8, P V bf16) is lossier than bf16 (median relerr ~1.7e-2 vs ~5e-3 on
    these peaked softmaxes) but bounded, also with K offsets (kcenter), and has no kattn_pack support."""
    c = A.resolve({"op": "attn", "target": "amx_bf16", "kv": kv, "qk": "int8"})
    ok, missing = A.runnable("amx_bf16")
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    so = A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror"))
    for sh in A.CHECK_SHAPES:
        cc = {**c, **A.PROBLEM, "pos0": -1, **sh, "threads": 3}
        for extra in ((), ("--k-bias", "100")):
            row = A.bench(so, cc, "hot", 0, ("--check-toks", "40", *extra), tol=6e-2)
            assert row["check"] == "ok", (sh, extra, row["relerr"])
    with pytest.raises(A.HarnessError):
        A.bench(so, {**c, "nq": 4, "nkv": 100, "threads": 1}, "hot", 0, ("--packed",))


def test_packed_kv_unsupported_engine():
    c = A.resolve({"op": "attn", "target": "avx512"})
    if not A.runnable("avx512")[0]:
        pytest.skip("host CPU lacks AVX-512")
    with pytest.raises(A.HarnessError):
        A.bench(A.build(c), {**c, "nq": 4, "nkv": 100, "threads": 1}, "hot", 0, ("--packed",))


@pytest.mark.parametrize("target", ["amx_bf16", "avx512"])
def test_mla_latent_attention(target):
    """MLA absorbed form: one shared 576-wide latent KV head, v = first 512 values of k."""
    _run({"target": target, "dk": 576, "dv": 512, "mla": 1, "kv": "f16"})
