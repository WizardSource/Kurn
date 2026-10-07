"""`layout composed`: packed layouts and kernels derived from the layout algebra
(kurn.layout) and the schedule lowering (kurn.sched) instead of a fixed template.

Keys (all neutral at their defaults, so every pre-existing config is unchanged; a
non-default value needs `layout composed`):

    lanes    0 (auto: 16 on AVX-512, 8 on AVX2) | 16 | 8      rows per vector
    plane    auto | kstep | khalf | rows | atom | none        bit-plane binding of sub-byte codes
    kblock   0 (auto: scale period) | 32 | 64 | 128 | 256     K values per record
    rgroup   1 | 2 | 4                                        lane groups interleaved into one record
    meta     auto | head | tail                               metadata placement in the record
    rgpad    0 | 64                                           extra bytes per row-group stride
    swizzle  0 | 1 | 2                                        XOR the vector index by row-group bits
    chains   0 (auto: 2) | 1 | 2 | 4                          int32 accumulator chains per lane group
    pfhint   t0 | t1 | t2 | nta                               prefetch locality hint
    pfgran   rec | line                                       prefetch one line per record or every line
    stages   1 | 2 | 3                                        prefetch cascade: stage i at prefetch * 2^i (far
                                                              stage with pfhint, nearer stages T1 / T0)
    kpanel   0 | 512 | 1024 | 2048                            K values per K panel (0: untiled)
    rpanel   0 | 4 | 16 | 64                                  row-group passes per row panel (with kpanel)
plus `align 64` and the algorithm values unpack=perm, correction=pair|float, scales=fold
(see kurn.sched). `auto` resolves to the values that reproduce the i16 / i8 layout.
"""

from .. import generic, hooks, sched

LAYOUT = "composed"
TARGETS = sched.LANE_TARGETS


def _possible(op, f, t):
    return t in TARGETS and f in generic.RECIPES and op in ("gemv", "verify") and sched.supports(generic.RECIPES[f])


def _key(values_fn, possible=None):
    possible = possible or _possible

    def legal(op, f, t):
        return values_fn(op, f, t) if possible(op, f, t) else values_fn.neutral

    return legal


def _pf_possible(op, f, t):
    """pfhint / pfgran: every recipe the row-interleaved lowerings take (i16 / i8 as well as composed)."""
    return t in TARGETS and f in generic.RECIPES and op in ("gemv", "verify")


def _vals(neutral, fn):
    fn.neutral = (neutral,)
    return fn


KEYS = {
    "lanes": (0, _vals(0, lambda op, f, t: (0, 16, 8) if t == "avx512_vnni" else (0, 8))),
    "plane": ("auto", _vals("auto", lambda op, f, t: ("auto",) + sched.planes_for(generic.RECIPES[f].bits))),
    "kblock": (0, _vals(0, lambda op, f, t: (0,) + tuple(k for k in (32, 64, 128, 256) if k % generic.RECIPES[f].period == 0))),
    "rgroup": (1, _vals(1, lambda op, f, t: (1, 2, 4))),
    "meta": ("auto", _vals("auto", lambda op, f, t: ("auto", "head", "tail"))),
    "rgpad": (0, _vals(0, lambda op, f, t: (0, 64))),
    "swizzle": (0, _vals(0, lambda op, f, t: (0, 1, 2))),
    "chains": (0, _vals(0, lambda op, f, t: (0, 1, 2, 4))),
    "pfhint": ("t0", _vals("t0", lambda op, f, t: ("t0", "t1", "t2", "nta"))),
    "pfgran": ("rec", _vals("rec", lambda op, f, t: ("rec", "line"))),
    "stages": (1, _vals(1, lambda op, f, t: (1, 2, 3))),
    "kpanel": (0, _vals(0, lambda op, f, t: (0, 512, 1024, 2048))),
    "rpanel": (0, _vals(0, lambda op, f, t: (0, 4, 16, 64))),
}
AUTO = {k: d for k, (d, _) in KEYS.items()}
PF_LAYOUTS = ("i16", "i8")  # the generic row-interleaved lowerings take pfgran / pfhint too
PF_KEYS = ("pfgran", "pfhint")

for _k, (_default, _fn) in KEYS.items():
    hooks.new_key(_k, _key(_fn, _pf_possible if _k in PF_KEYS else None), _default)
hooks.AUTO_VALUES.update(lanes=0, kblock=0, chains=0)
hooks.ENUM_KEYS.update(KEYS)


def _layout_value(op, f, t):
    return (LAYOUT,) if _possible(op, f, t) else ()


def _algo_values(key):
    def extra(op, f, t):
        return sched.algo_values(generic.RECIPES[f], t)[key] if _possible(op, f, t) else ()

    return extra


hooks.extra_values("layout", _layout_value)
for _k in ("unpack", "correction", "scales", "accum"):
    hooks.extra_values(_k, _algo_values(_k))
hooks.extra_values("align", lambda op, f, t: (64,) if _possible(op, f, t) else ())


def fill(c):
    """Composed config with every `auto` replaced (does not modify c)."""
    r = generic.RECIPES[c["weights"]]
    d = sched.defaults(r, c["target"])
    out = dict(c)
    for k in ("lanes", "kblock", "chains"):
        if out[k] == 0:
            out[k] = d[k]
    for k in ("plane", "meta", "unpack", "correction", "scales", "accum"):
        if out[k] == "auto":
            out[k] = d[k]
    if out["align"] == "packed":
        out["align"] = 0
    return out


def _reason(c):
    if c["layout"] != LAYOUT:
        return None
    try:
        return sched.check(fill(c))
    except (KeyError, ValueError) as e:
        return str(e)


hooks.EXTRA_INVALID.extend(
    [
        (
            lambda c: c["layout"] != LAYOUT and any(c[k] != AUTO[k] for k in KEYS if not (c["layout"] in PF_LAYOUTS and k in PF_KEYS)),
            "lanes/plane/kblock/rgroup/meta/rgpad/swizzle/chains/stages/kpanel/rpanel apply to layout=composed only "
            "(pfhint/pfgran: composed, i16 and i8)",
        ),
        (lambda c: c["layout"] == LAYOUT and c["act"] != "once", "act applies to layout=native only (use act=once)"),
        (lambda c: _reason(c) is not None, "illegal composed layout/schedule (run `kurn check` with the config for the reason)"),
    ]
)


def _resolve(c):
    if c["layout"] == LAYOUT:
        f = fill(c)
        for k in ("lanes", "kblock", "chains", "plane", "meta", "unpack", "correction", "scales", "accum"):
            c[k] = f[k]


hooks.RESOLVE_HOOKS.append(_resolve)


def _lower(target, c):
    return sched.lower(target, fill(c))


hooks.LOWERINGS[LAYOUT] = _lower


# --------------------------------------------------------------------------- covering enumeration
def covering(op, f, t):
    """Override dicts covering every composed key value at least once (one-at-a-time from
    the defaults, plus the algorithm combinations that need each other)."""
    from ..toolchain import cpu_flags

    r = generic.RECIPES[f]
    base = {"layout": LAYOUT, "act": "once", "prefetch": 0}
    if op == "verify":
        base.update(rows=1, cols=4)
        yield base
        for cols in (2, 8):
            yield {**base, "cols": cols}
        yield {**base, "rows": 2, "cols": 2, "rgroup": 2}
        yield {**base, "cols": 8, "kpanel": 1024, "rpanel": 16}
        yield {**base, "prefetch": 8, "pfgran": "line", "lanes": 8 if t == "avx512_vnni" else 0}
        return
    base["rows"] = 2
    yield base
    yield {**base, "rows": 1}
    yield {**base, "rows": 4, "rgroup": 4}
    for k, (_, fn) in KEYS.items():
        for v in fn(op, f, t)[1:]:
            ov = {**base, k: v}
            if k == "rgroup":
                ov["rows"] = max(2, v)
            if k in ("pfhint", "pfgran", "stages"):
                ov["prefetch"] = 8
            if k == "kpanel":
                ov["rpanel"] = 4
            if k == "rpanel":
                ov["kpanel"] = 512 if r.period <= 512 else 0
            if k == "swizzle":
                ov.update(rows=4, rgroup=4)
            yield ov
    yield {**base, "align": 64}
    yield {**base, "prefetch": 8}
    algo = sched.algo_values(r, t)
    for u in algo["unpack"]:
        if u != "perm" or "avx512vbmi" in cpu_flags():  # vpermb: Ice Lake and later
            yield {**base, "unpack": u}
    for corr in algo["correction"]:
        ov = {**base, "correction": corr}
        if corr == "float":
            ov.update(scales="fold", accum="float")
        if corr == "pair":
            ov.update(scales="unpacked")
        yield ov
    for s in algo["scales"]:
        ov = {**base, "scales": s}
        if s == "fold":
            ov.update(correction="float", accum="float")
        yield ov
    for a in algo["accum"]:
        yield {**base, "accum": a}
    if r.bits < 8:
        yield {**base, "plane": "rows", "rows": 1, "kblock": 2 * r.period if r.period < 256 else 0}


hooks.ENUMERATORS[LAYOUT] = covering

hooks.GOLDEN.update(
    {
        "composed_q4_K_gemv_avx512": dict(
            op="gemv", weights="q4_K", target="avx512_vnni", layout=LAYOUT, rows=2, correction="pair", unpack="perm", rgroup=2
        ),
        "composed_q4_0_vfy_avx2": dict(
            op="verify", weights="q4_0", target="avx2_vnni", layout=LAYOUT, rows=1, cols=4, plane="khalf", kblock=64
        ),
    }
)
