"""Split-bf16 Q K^T (codegen key qk_hi) in the bf16 tile engines.

Plain bf16 rounds q and k to 8 mantissa bits; on the harness data its worst-of-10-seeds error is 0.8-1.5e-2
at 1K-32K keys, against the 1e-2 tolerance. Query rows that see >= qk_hi keys add q_lo k_hi + q_hi k_lo
(hi = bf16(x), lo = bf16(x - hi)), which leaves the P V rounding (~3e-3) as the error floor. The other rows
of a tile multiply zeros in the extra terms: their output is bit-identical to qk_hi 0, so a row's result
depends on its own key count only (also across n_kv padding and batches)."""

import ctypes
import threading

import pytest

from kurn import attention as A
from kurn.spec import SpecError

np = pytest.importorskip("numpy")

SPLIT_TOL = 5e-3  # split rows: the P V rounding floor is 1.5-3.2e-3 on the check shapes


def _need(target):
    ok, missing = A.runnable(target)
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")


def test_qk_hi_resolve():
    assert A.resolve({"target": "amx_bf16"})["qk_hi"] == 4096
    assert A.resolve({"target": "avx512_bf16"})["qk_hi"] == 4096
    assert A.resolve({"target": "avx512"})["qk_hi"] == 0  # f32 scores already (and exact mode)
    assert A.resolve({"target": "amx_bf16", "qk": "int8"})["qk_hi"] == 0
    for bad in ({"target": "avx512", "qk_hi": 4096}, {"target": "amx_bf16", "qk_hi": 3000},
                {"target": "amx_bf16", "qk": "int8", "qk_hi": 1}):  # fmt: skip
        with pytest.raises(SpecError):
            A.resolve(bad)
    assert "#define KA_QKHI 4096" in A.generate(A.resolve({"target": "amx_bf16"}))


SPLIT_CFGS = [
    {"target": "amx_bf16", "kv": "f16"},
    {"target": "amx_bf16", "kv": "q8_0", "tile_q": 128, "tile_kv": 256},
    {"target": "amx_bf16", "kv": "bf16", "dk": 64, "tile_q": 16, "tile_kv": 64},
    {"target": "amx_bf16", "kv": "k4c_q4"},
    {"target": "amx_bf16", "kv": "k4c_q8", "dec_rows": 0},
    {"target": "avx512_bf16", "kv": "f16"},
    {"target": "avx512_bf16", "kv": "q8_0", "dec_rows": 0},
    {"target": "amx_bf16", "kv": "q8_0", "dk": 256, "tile_kv": 64},
    {"target": "amx_bf16", "kv": "f16", "dk": 576, "dv": 512, "mla": 1},
]


@pytest.mark.parametrize("cfg", SPLIT_CFGS, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
def test_split_every_row_on_check_shapes(cfg):
    """qk_hi 1 (every row split) on the awkward shapes: within the P V floor, every format and schedule."""
    c = A.resolve({"op": "attn", **cfg, "qk_hi": 1})
    _need(c["target"])
    row = A.check(A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror")), c)
    assert row["check"] == "ok" and row["relerr"] < SPLIT_TOL, row


@pytest.mark.parametrize(
    "cfg",
    [{"target": "amx_bf16", "kv": "f16"}, {"target": "avx512_bf16", "kv": "q8_0"},
     {"target": "amx_bf16", "kv": "f16", "dk": 576, "dv": 512, "mla": 1}],  # fmt: skip
    ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()),
)
def test_split_packed_kv(cfg):
    """kattn_pack also stores k_lo (after V); packed split tiles read it in place, with the packed K mean."""
    mla = cfg.pop("mla", 0)
    c = A.resolve({"op": "attn", **cfg, "qk_hi": 1})
    _need(c["target"])
    so = A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror"))
    for sh in A.CHECK_SHAPES:
        cc = {**c, **A.PROBLEM, "pos0": -1, **sh, "threads": 3}
        if mla:
            cc.update(mla=1, kv_heads=1)
        for extra in ((), ("--packed-split", "33"), ("--k-bias", "100", "--packed-split", "7")):
            row = A.bench(so, cc, "hot", 0, ("--check-toks", "40", "--packed", *extra))
            assert row["check"] == "ok" and row["relerr"] < SPLIT_TOL, (sh, extra, row["relerr"])


@pytest.mark.parametrize("target", ["amx_bf16", "avx512_bf16"])
@pytest.mark.parametrize(
    "case",
    [("prefill-8B-d32768", {"nq": 512, "nkv": 32768, "heads": 32, "kv_heads": 8, "dk": 128}),
     ("mla-h16-d8192", {"nq": 1, "nkv": 8192, "heads": 16, "kv_heads": 1, "dk": 576, "mla": 1})],  # fmt: skip
    ids=lambda c: c[0],
)
def test_long_context_checks(target, case):
    """The attention matrix's two long-context bf16 checks (benchmarks/v0.2/attn/matrix.py, prefill 8B shape at
    32K and MLA decode h16 at 8K) at the default qk_hi and the 1e-2 tolerance. Plain bf16 gave 1.43e-2 and
    1.23e-2 at seed 1 here; the split gives 2.9e-3 and 3.0e-3."""
    _, prob = case
    c = A.resolve({"target": target, "kv": "f16", "causal": 1, **prob, "threads": 8})
    _need(target)
    so = A.build(c)
    for seed in (1, 2, 3):
        row = A.bench(so, c, "hot", 0, ("--check-toks", "4", "--seed", str(seed)))
        assert row["check"] == "ok" and row["relerr"] < SPLIT_TOL, (seed, row["relerr"])


# ---------------------------------------------------------------- in-process (ctypes)


class Args(ctypes.Structure):
    _fields_ = [
        ("n_q", ctypes.c_int64), ("n_kv", ctypes.c_int64), ("q_pos0", ctypes.c_int64),
        ("n_head", ctypes.c_int32), ("n_head_kv", ctypes.c_int32), ("causal", ctypes.c_int32), ("scale", ctypes.c_float),
        ("q", ctypes.c_void_p), ("q_s_tok", ctypes.c_int64), ("q_s_head", ctypes.c_int64),
        ("k", ctypes.c_void_p), ("k_s_tok", ctypes.c_int64), ("k_s_head", ctypes.c_int64),
        ("v", ctypes.c_void_p), ("v_s_tok", ctypes.c_int64), ("v_s_head", ctypes.c_int64),
        ("mask", ctypes.c_void_p), ("mask_s_tok", ctypes.c_int64),
        ("out", ctypes.c_void_p), ("o_s_tok", ctypes.c_int64), ("o_s_head", ctypes.c_int64),
        ("k_tail", ctypes.c_void_p), ("kt_s_tok", ctypes.c_int64), ("kt_s_head", ctypes.c_int64),
        ("rope_freq", ctypes.c_void_p), ("k_pos0", ctypes.c_int64), ("rope_dim", ctypes.c_int32), ("rope_mode", ctypes.c_int32),
    ]  # fmt: skip


def _lib(target, flags=(), **kw):
    _need(target)
    lib = ctypes.CDLL(A.build(A.resolve({"target": target, "kv": "f16", **kw}), extra_flags=flags))
    lib.kattn_workspace.argtypes, lib.kattn_workspace.restype = [ctypes.POINTER(Args), ctypes.c_int], ctypes.c_size_t
    lib.kattn.argtypes, lib.kattn.restype = [ctypes.POINTER(Args), ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None
    return lib


def _run(lib, q, k, v, n_kv, pos0, mask=None, nth=3):
    """Attention of q [n_q][nh][d] over the first n_kv rows of k, v: causal from pos0, or the fp16 mask rows."""
    nq, nh, d = q.shape
    out = np.full((nq, nh, d), np.nan, np.float32)
    a = Args(nq, n_kv, pos0, nh, k.shape[1], 0 if mask is not None else 1, 1 / np.sqrt(d),
             q.ctypes.data, nh * d, d, k.ctypes.data, k.strides[0], k.strides[1], v.ctypes.data, v.strides[0], v.strides[1],
             None if mask is None else mask.ctypes.data, 0 if mask is None else mask.shape[1],
             out.ctypes.data, nh * d, d)  # fmt: skip
    ws = np.zeros(lib.kattn_workspace(ctypes.byref(a), nth) + 64, np.uint8)
    ts = [threading.Thread(target=lib.kattn, args=(ctypes.byref(a), ws.ctypes.data, i, nth)) for i in range(nth)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def _problem(seed, nq, nkv, nh=6, nhkv=2, d=128):
    rng = np.random.default_rng(seed)
    q = (3 * rng.standard_normal((nq, nh, d))).astype(np.float32)
    k = rng.standard_normal((nkv, nhkv, d)).astype(np.float16)
    v = rng.standard_normal((nkv, nhkv, d)).astype(np.float16)
    return q, k, v


def _mask(nq, pos0, n_kv):
    """ggml-style causal mask (0 / -inf), rows padded to 64."""
    m = np.full((nq, (n_kv + 63) // 64 * 64), np.float16(-np.inf), np.float16)
    for t in range(nq):
        m[t, : pos0 + t + 1] = 0
    return m


T = 2048  # threshold for the in-process tests; rows of token t see pos0 + t + 1 keys
NQ, POS0, NKV_PAD = 100, 2000, 2304  # tokens 0-46 see < T keys, 47-99 see >= T


@pytest.mark.parametrize("target", ["amx_bf16", "avx512_bf16"])
@pytest.mark.parametrize("masked", [0, 1], ids=["causal", "mask"])
def test_rows_follow_their_own_key_count(target, masked):
    """In one call, rows below the threshold are bit-identical to qk_hi 0 and rows at or above it to qk_hi 1,
    with G = 3 so that AMX 32-row and AVX-512 4-row blocks mix split and plain rows. With the mask, n_kv is
    padded past the last visible key (as llama.cpp's KV view is) and does not change a row's key count."""
    q, k, v = _problem(5, NQ, NKV_PAD)
    n_kv, mask = (NKV_PAD, _mask(NQ, POS0, NKV_PAD)) if masked else (POS0 + NQ, None)
    out = {h: _run(_lib(target, qk_hi=h), q, k, v, n_kv, POS0, mask) for h in (T, 0, 1)}
    split = np.array([POS0 + t + 1 >= T for t in range(NQ)])
    assert split.any() and not split.all()
    assert np.array_equal(out[T][~split], out[0][~split])
    assert np.array_equal(out[T][split], out[1][split])
    assert not np.array_equal(out[0][split], out[1][split])


@pytest.mark.parametrize("target", ["amx_bf16", "avx512_bf16"])
def test_batch_invariance_across_threshold(target):
    """Tile engine with one KV split and no decode engine (what exact mode fixes, here on the bf16 engines): a
    token's output is the same, bit for bit, in a 100-token batch with n_kv 2304 and alone with n_kv padded to
    256 above its own keys, on both sides of the threshold. On AMX the lone token (one 32-row block) takes the
    transposed split path and the batch the row-tile one."""
    lib = _lib(target, dec_rows=0, split=1, qk_hi=T)
    q, k, v = _problem(9, NQ, NKV_PAD)
    batch = _run(lib, q, k, v, NKV_PAD, POS0, _mask(NQ, POS0, NKV_PAD))
    for t in (0, 46, 47, 48, 99):
        n_kv = (POS0 + t + 1 + 255) // 256 * 256
        one = _run(lib, np.ascontiguousarray(q[t : t + 1]), k, v, n_kv, POS0 + t, _mask(1, POS0 + t, n_kv), nth=2)
        assert np.array_equal(one[0], batch[t]), t
