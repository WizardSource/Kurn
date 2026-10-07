"""layout=composed (kurn.ext.layout + kurn.sched): the layout-algebra lowering
reproduces the fixed layouts bit for bit, and random composed layouts/schedules
compute the same results as the reference."""

import ctypes
import random

import re

import pytest

from kurn import generic, harness, sched, spec, toolchain
from kurn.ext import layout as ext

from conftest import BLOCKS

TARGET_LAYOUT = {"avx512_vnni": "i16", "avx2_vnni": "i8"}


def _need(t, c=None):
    mode, why = toolchain.run_mode(t)
    if mode != "native":
        pytest.skip(why)
    if c and c["unpack"] == "perm" and "avx512vbmi" not in toolchain.cpu_flags():
        pytest.skip("unpack=perm needs AVX-512 VBMI")


def _algo_cases():
    for f, r in generic.RECIPES.items():
        if not sched.supports(r):
            continue
        for t in TARGET_LAYOUT:
            legal, comp = generic.legal_keys(f, t), sched.algo_values(r, t)
            for corr in (v for v in legal["correction"] if v in comp["correction"]):
                for sc in (v for v in legal["scales"] if v in comp["scales"]):
                    for u in (v for v in legal["unpack"] if v in comp["unpack"]):
                        base = dict(op="gemv", weights=f, target=t, rows=1, correction=corr, scales=sc, unpack=u)
                        try:  # only combinations both lowerings accept (e.g. weight + perm is legal in neither)
                            spec.resolve({**base, "layout": TARGET_LAYOUT[t]})
                            spec.resolve({**base, "layout": "composed"})
                        except spec.SpecError:
                            continue
                        yield f, t, corr, sc, u


def _packed(c, w, K, N):
    """Run <entry>_prepare and return (packed_t pointer, buffer address)."""
    lib = ctypes.CDLL(toolchain.build(c))
    from kurn.kernels import kernel

    prep = getattr(lib, kernel(c).entry + "_prepare")
    prep.restype = ctypes.c_void_p
    prep.argtypes = (ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64)
    pk = prep(w, K, N)
    return lib, ctypes.c_void_p.from_address(pk + 16).value


def _bytes(addr, n):
    return ctypes.string_at(addr, n)


@pytest.mark.parametrize("f,t,corr,sc,u", list(_algo_cases()))
def test_composed_defaults_reproduce_i16_i8_bit_identical(f, t, corr, sc, u):
    _need(t)
    r = generic.RECIPES[f]
    rng = random.Random(7)
    K, N = 2 * max(256, r.block), 48
    w = BLOCKS[f](rng, N * K // r.block)
    base = dict(op="gemv", weights=f, target=t, rows=1, correction=corr, scales=sc, unpack=u)
    old = spec.resolve({**base, "layout": TARGET_LAYOUT[t]})
    new = spec.resolve({**base, "layout": "composed"})
    lay = sched.build_layout(ext.fill(new))
    n = (N // lay.rows) * (K // lay.kblock) * lay.rec_bytes
    _, a = _packed(old, w, K, N)
    _, b = _packed(new, w, K, N)
    assert _bytes(a, n) == _bytes(b, n)


@pytest.mark.parametrize("align", ["packed", 64])
def test_composed_reproduces_vnni16_bit_identical(align):
    _need("avx512_vnni")
    rng = random.Random(3)
    K, N = 512, 64
    w = BLOCKS["q8_0"](rng, N * K // 32)
    base = dict(op="gemv", weights="q8_0", target="avx512_vnni", rows=1, align=align)
    old = spec.resolve({**base, "layout": "vnni16"})
    new = spec.resolve({**base, "layout": "composed", **sched.preset_config("vnni16", base)})
    lay = sched.build_layout(ext.fill(new))
    assert lay.rec_bytes == (576 if align == 64 else 544)
    n = (N // 16) * (K // 32) * lay.rec_bytes
    _, a = _packed(old, w, K, N)
    _, b = _packed(new, w, K, N)
    assert _bytes(a, n) == _bytes(b, n)


def test_neutral_defaults_keep_existing_configs():
    old = list(spec.legal_configs())
    assert all(c["layout"] != "composed" for c in old)
    for c in old:
        assert all(c[k] == ext.AUTO[k] for k in ext.KEYS)
    new = list(spec.legal_configs(enumerated=True))
    assert [c for c in new if c["layout"] != "composed"] == old


COVERING = [c for c in spec.legal_configs(enumerated=True) if c["layout"] == "composed"]
_KEYS = ("op", "weights", "target", "rows", "cols", "prefetch", "unpack", "correction", "scales", "accum", "align") + tuple(ext.KEYS)


def _id(c):
    return "-".join(str(c[k]) for k in _KEYS)


@pytest.mark.parametrize("c", COVERING, ids=_id)
def test_covering_numerics(c):
    _need(c["target"], c)
    row = harness.check(toolchain.build(c), c)
    assert row["check"] == "ok", row


@pytest.mark.parametrize("c", COVERING, ids=_id)
def test_covering_compiles_without_warnings(c):
    from kurn.kernels import generate

    toolchain.compile_source(generate(c), c["target"], stem="warn", extra_flags=("-Wall", "-Wextra", "-Wshadow", "-Werror"), shared=False)


@pytest.mark.parametrize("key", sorted(ext.KEYS))
def test_composed_keys_rejected_elsewhere(key):
    _, fn = ext.KEYS[key]
    v = fn("gemv", "q4_0", "avx512_vnni")[1]
    # pfgran / pfhint also apply to i16 / i8: check them on a layout that has neither (q8_0 native)
    w, layout = ("q8_0", "native") if key in ext.PF_KEYS else ("q4_0", "i16")
    with pytest.raises(spec.SpecError, match="layout=composed only"):
        spec.resolve(dict(op="gemv", weights=w, target="avx512_vnni", layout=layout, **{key: v}))


@pytest.mark.parametrize("fmt", ["q8_0", "q4_K", "q4_0", "mxfp4"])
def test_i16_pfgran_line_prefetches_every_record_line(fmt):
    from kurn.kernels import generate

    base = dict(op="verify", weights=fmt, target="avx512_vnni", layout="i16", cols=3, rows=2, prefetch=4)
    one = generate(spec.resolve(dict(base)))
    every = generate(spec.resolve(dict(base, pfgran="line", pfhint="t1")))
    rec = int(re.search(r"#define REC_BYTES (\d+)", every).group(1))
    assert one.count("_mm_prefetch") == 2  # one line per record and row group
    assert every.count("_MM_HINT_T1") >= 2 * (rec // 64) or "kg * KG_BYTES" in every
    assert every.count("_mm_prefetch") > one.count("_mm_prefetch")


def test_covering_set_covers_every_value():
    cs = COVERING
    for (op, f), targets in spec.TARGETS.items():
        for t in targets:
            if not ext._possible(op, f, t) or op != "gemv":
                continue
            mine = [c for c in cs if (c["op"], c["weights"], c["target"]) == (op, f, t)]
            for k, (_, fn) in ext.KEYS.items():
                vals = {c[k] for c in mine}
                legal = set()
                for v in fn(op, f, t):
                    try:
                        legal.add(spec.resolve({**dict(op=op, weights=f, target=t, layout="composed", rows=4, rgroup=4), k: v})[k])
                    except spec.SpecError:
                        pass
                assert legal <= vals, (op, f, t, k, legal - vals)


def _random_configs(n, seed=11):
    rng = random.Random(seed)
    keys = list(ext.KEYS) + ["unpack", "correction", "scales", "accum", "rows", "prefetch", "align"]
    out, tries = [], 0
    triples = [(op, f, t) for (op, f), ts in spec.TARGETS.items() for t in ts if ext._possible(op, f, t)]
    while len(out) < n and tries < 50 * n:
        tries += 1
        op, f, t = rng.choice(triples)
        c = {"op": op, "weights": f, "target": t, "layout": "composed"}
        if op == "verify":
            c["cols"] = rng.choice(spec.SCHEDULE["cols"](op, f, t))
        for k in keys:
            if rng.random() < 0.5:
                c[k] = rng.choice(spec.SCHEDULE[k](op, f, t))
        try:
            out.append(spec.resolve(c))
        except spec.SpecError:
            continue
    return out


@pytest.mark.parametrize("c", _random_configs(48), ids=_id)
def test_random_composed_numerics(c):
    _need(c["target"], c)
    row = harness.check(toolchain.build(c), c)
    assert row["check"] == "ok", row


def test_illegal_composed_reasons():
    base = dict(op="gemv", weights="q4_K", target="avx512_vnni", layout="composed")
    c = ext.fill({**spec.resolve(base), "correction": "pair", "scales": "packed"})
    assert "pair" in sched.check(c)
    c = ext.fill({**spec.resolve(base), "plane": "atom"})
    assert "plane=atom" in sched.check(c)
    with pytest.raises(spec.SpecError):
        spec.resolve({**base, "kblock": 128})  # q4_K period is 256


@pytest.mark.parametrize(
    "op,f,t",
    [("gemv", "q4_0", "avx512_vnni"), ("verify", "q8_0", "avx512_vnni"), ("verify", "q4_K", "avx2_vnni"), ("gemv", "q1_0", "avx2_vnni")],
)
def test_tiling_ragged_k_panels(op, f, t):
    _need(t)
    c = spec.resolve(
        dict(
            op=op,
            weights=f,
            target=t,
            layout="composed",
            rows=1,
            kpanel=512,
            rpanel=4,
            prefetch=4,
            stages=3,
            **({"cols": 8} if op == "verify" else {}),
        )
    )
    extra = ["--N", "200", "--K", "1280" if f != "q4_K" else "1536"] + (["--M", "7"] if op == "verify" else [])
    row = harness.check(toolchain.build(c), c, extra=extra)
    assert row["check"] == "ok", row
