"""Spec parsing and validation.

A spec is a flat file of `key value` lines (`#` starts a comment). Every key has
a closed set of legal values that depends on (op, weights, target); an illegal
value fails with the list of legal ones. A `tune` line declares a search space:

    kernel   q8_0_gemv
    op       gemv
    weights  q8_0
    target   avx512_vnni
    layout   vnni16
    tune     rows=1,2,4,8 prefetch=0,8,16 threads=4,6,8
"""

import itertools
import os

from . import generic, hooks
from .formats import FORMATS
from .kernels import KERNELS

MAX_THREADS = 256

# Derived from the kernel registry (kernels.py): op -> weight formats, (op, weights) -> targets.
OPS = {op: tuple(w for o, w in KERNELS if o == op) for op in dict.fromkeys(o for o, _ in KERNELS)}
TARGETS = {key: k.targets for key, k in KERNELS.items()}

THREADS = tuple(range(1, MAX_THREADS + 1))


GENERIC_LAYOUT = {"avx512_vnni": "i16", "avx2_vnni": "i8"}


def _generic_possible(op, f, t):
    return t in GENERIC_LAYOUT and f in generic.RECIPES and op in ("gemv", "verify")


def _native_only(op, f):
    return (op, f) not in (("gemv", "q8_0"), ("gemv", "q4_K"), ("gemm", "q8_0"))


def _layout(op, f, t):
    out = () if (_native_only(op, f) and t != "scalar") else ("native",)
    if _vnni16_capable(op, f, t):
        out += ("vnni16",)
    if _generic_possible(op, f, t):
        out += (GENERIC_LAYOUT[t],)
        if t == "avx512_vnni" and op == "gemv" and generic.RECIPES[f].bits <= 2:
            out += ("l32",)  # T-MAC-style lookup tables instead of multiplies
    return out or ("native",)


def _algo(key):
    def allowed(op, f, t):
        out = ("auto",)
        if _generic_possible(op, f, t):
            out += generic.legal_keys(f, t)[key]
        if _native_q4k_algo(op, f, t):
            out += tuple(v for v in NATIVE_Q4K_ALGO.get(key, ()) if v not in out)
        return out

    return allowed


def _algo_ok(c):
    """Algorithm keys must be legal for the chosen layout (the SCHEDULE sets are unions over layouts)."""
    if c["layout"] == "l32":
        legal = generic.legal_keys(c["weights"], c["target"])
        return (
            c["unpack"] == "auto"
            and c["correction"] in ("auto", "act")
            and c["scales"] in ("auto", "unpacked")
            and c["accum"] in ("auto",) + legal["accum"]
        )
    if c["layout"] in ("i16", "i8"):
        legal = generic.legal_keys(c["weights"], c["target"])
        return all(c[k] in ("auto",) + legal[k] for k in legal)
    if c["layout"] == "composed":
        return True  # kurn.ext.layout checks the composed combinations (sched.check)
    if _native_q4k_algo(c["op"], c["weights"], c["target"]) and c["layout"] == "native":
        return c["unpack"] == "auto" and all(c[k] in ("auto",) + v for k, v in NATIVE_Q4K_ALGO.items())
    return all(c[k] == "auto" for k in ("unpack", "correction", "scales", "accum"))


def _rows(op, f, t):
    if _generic_possible(op, f, t) and (_native_only(op, f) or op == "verify"):
        return (1, 2, 4)
    if t == "amx":
        return (1, 2)
    if op == "gemm":
        return (2, 4, 6)
    return (1, 2, 4) if f == "q4_K" else (1, 2, 4, 8)


def _cols(op, f, t):
    if op == "verify":
        return (2, 4, 8)
    if t == "amx":
        return (1, 2)
    return (2, 4, 6) if op == "gemm" else (1,)


NATIVE_Q4K_ALGO = {"correction": ("act", "scalar"), "scales": ("kmask", "bytewise"), "accum": ("float", "int")}


def _native_q4k_algo(op, f, t):
    return (op, f, t) == ("gemv", "q4_K", "avx512_vnni")


def _act(op, f, t):
    if _native_q4k_algo(op, f, t):
        return ("once", "inline")
    if t == "avx2" and f == "q8_0":
        return ("inline",)
    if f == "q8_0" and op == "gemv" and t not in ("scalar", "neon"):
        return ("once", "inline")
    return ("once",)


def _vnni16_capable(op, f, t):
    return (op, f, t) == ("gemv", "q8_0", "avx512_vnni")


def _i16_q8_rows8(op, f, t):
    """The Q8_0 GEMV on the i16 layout takes vnni16's rows=8 (8 weight streams per core; same records,
    so the verify kernels still share the packing). vnni16's align=64 is not offered: the header
    padding adds 6% bytes and measured slower from DRAM (benchmarks/v0.2/q4fix)."""
    return (op, f, t) == ("gemv", "q8_0", "avx512_vnni")


# Legal values per schedule/runtime key, as a function of (op, weights, target).
SCHEDULE = {
    "layout": _layout,
    "unpack": _algo("unpack"),
    "correction": _algo("correction"),
    "scales": _algo("scales"),
    "accum": _algo("accum"),
    "align": lambda op, f, t: ("packed", 64) if _vnni16_capable(op, f, t) else ("packed",),
    "rows": _rows,
    "cols": _cols,
    "act": _act,
    "prefetch": lambda op, f, t: (0,) if op == "gemm" or t == "scalar" else (0, 2, 4, 8, 16, 32),
    "threads": lambda op, f, t: THREADS,
    "wait": lambda op, f, t: ("spin", "sleep"),
}

DEFAULTS = {
    "unpack": "auto",
    "correction": "auto",
    "scales": "auto",
    "accum": "auto",
    "align": "packed",
    "layout": "native",
    "rows": 4,
    "cols": 4,
    "act": "once",
    "prefetch": 0,
    "threads": min(8, os.cpu_count() or 8),
    "wait": "spin",
}

# Keys that change the generated C (everything else is a runtime knob).
CODEGEN_KEYS = ("op", "weights", "target", "layout", "align", "rows", "cols", "act", "prefetch", "unpack", "correction", "scales", "accum")

INVALID_COMBOS = [
    (
        lambda c: c["op"] == "gemm" and c["target"] == "avx512_vnni" and c["rows"] * c["cols"] > 24,
        "register tile rows*cols exceeds 24 accumulators for avx512_vnni",
    ),
    (lambda c: c["layout"] == "vnni16" and c["act"] != "once", "act applies to layout=native only (use act=once)"),
    (
        lambda c: c["layout"] not in ("vnni16", "composed") and c["align"] != "packed",
        "align applies to layout=vnni16 only (and layout=composed)",
    ),
    (lambda c: c["layout"] in ("i16", "i8") and c["act"] != "once", "act applies to layout=native only (use act=once)"),
    (lambda c: c["layout"] == "vnni16" and c["act"] != "once", "act applies to layout=native only (use act=once)"),
    (
        lambda c: not _algo_ok(c),
        "this unpack/correction/scales/accum value is not legal for the chosen layout (see `kurn check` for the per-layout sets)",
    ),
    (lambda c: c["layout"] in ("i16", "i8") and c["rows"] * c["cols"] > 8, "rows * cols must be <= 8 for i16/i8"),
    (
        lambda c: (
            c["layout"] in ("i16", "i8")
            and c["rows"] not in (1, 2, 4)
            and not (c["rows"] == 8 and _i16_q8_rows8(c["op"], c["weights"], c["target"]))
        ),
        "rows must be 1, 2 or 4 for i16/i8 (8 for the Q8_0 GEMV on avx512_vnni)",
    ),
    (lambda c: c["layout"] == "l32" and c["rows"] not in (1, 2), "rows must be 1 or 2 for l32 (32-row groups)"),
]


def _with_extras(key, fn):
    def allowed(op, f, t):
        out = tuple(fn(op, f, t))
        for extra in hooks.EXTRA_VALUES.get(key, ()):
            out += tuple(v for v in extra(op, f, t) if v not in out)
        return out

    return allowed


# kurn.ext modules were imported with kurn.kernels above, so the hook tables are complete here.
for _k, (_fn, _default, _codegen) in hooks.NEW_KEYS.items():
    SCHEDULE[_k], DEFAULTS[_k] = _fn, _default
    if _codegen:
        CODEGEN_KEYS += (_k,)
SCHEDULE = {k: _with_extras(k, fn) for k, fn in SCHEDULE.items()}
INVALID_COMBOS += hooks.EXTRA_INVALID
RUNTIME_KEYS = ("threads", "wait")

KNOWN_KEYS = ("kernel", "op", "weights", "target") + tuple(SCHEDULE)


class SpecError(Exception):
    pass


def _coerce(v):
    try:
        return int(v)
    except ValueError:
        return v


def parse(text):
    """Parse spec text into (spec dict, tune space dict). Does not validate values."""
    spec, tune = {}, {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition(" ")
        val = val.strip()
        if not val:
            raise SpecError(f"line {n}: {key!r} has no value")
        if key == "tune":
            for item in val.split():
                k, _, vs = item.partition("=")
                if not vs:
                    raise SpecError(f"line {n}: tune entries look like key=v1,v2 (got {item!r})")
                if k in tune:
                    raise SpecError(f"line {n}: duplicate tune key {k!r}")
                tune[k] = [_coerce(v) for v in vs.split(",")]
        else:
            if key in spec:
                raise SpecError(f"line {n}: duplicate key {key!r}")
            if len(val.split()) != 1:
                raise SpecError(f"line {n}: {key!r} takes one value (got {val!r})")
            spec[key] = _coerce(val)
    return spec, tune


def load(path):
    """Read and parse a spec file."""
    with open(path) as fh:
        return parse(fh.read())


def parse_overrides(items):
    """`k=v` strings -> (single overrides, list overrides for `k=v1,v2`)."""
    single, lists = {}, {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k or not v:
            raise SpecError(f"override {item!r}: expected key=value")
        if "," in v:
            lists[k] = [_coerce(x) for x in v.split(",")]
        else:
            single[k] = _coerce(v)
    return single, lists


def _allowed_str(k, allowed):
    if k == "threads":
        return f"an integer in 1..{MAX_THREADS}"
    return f"one of {list(allowed)}"


def resolve(spec, overrides=None):
    """Validate a spec (plus overrides) and fill defaults. Returns the full config dict."""
    c = dict(spec)
    c.update(overrides or {})
    for req in ("op", "weights", "target"):
        if req not in c:
            raise SpecError(f"missing required key {req!r}")
    op, fmt, tgt = c["op"], c["weights"], c["target"]
    if op not in OPS:
        raise SpecError(f"op {op!r}: expected one of {list(OPS)}")
    if fmt not in OPS[op]:
        raise SpecError(f"weights {fmt!r} not supported for op {op}: expected one of {list(OPS[op])}")
    if tgt not in TARGETS[(op, fmt)]:
        raise SpecError(f"target {tgt!r} not supported for {op}/{fmt}: expected one of {list(TARGETS[(op, fmt)])}")
    for k in c:
        if k not in KNOWN_KEYS:
            raise SpecError(f"unknown key {k!r}: expected one of {sorted(KNOWN_KEYS)}")
    if op == "verify" and "rows" not in c:
        c["rows"] = 1  # rows * cols <= 8; one row group keeps every verify width legal
    for k, allowed_fn in SCHEDULE.items():
        allowed = allowed_fn(op, fmt, tgt)
        if k not in c:
            c[k] = DEFAULTS[k] if DEFAULTS[k] in allowed else allowed[0]
        if c[k] not in allowed:
            raise SpecError(f"{k}={c[k]!r} not allowed for {op}/{fmt}/{tgt}: expected {_allowed_str(k, allowed)}")
    for bad, msg in INVALID_COMBOS:
        if bad(c):
            raise SpecError(msg)
    if c["layout"] == "native" and _native_q4k_algo(op, fmt, tgt):  # auto -> the hand-written v2 choices
        for k, v in NATIVE_Q4K_ALGO.items():
            if c[k] == "auto":
                c[k] = v[0]
    if c["layout"] == "l32":
        c["unpack"], c["correction"], c["scales"] = "lut", "act", "unpacked"
        if c["accum"] == "auto":
            c["accum"] = "float"
    if c["layout"] in ("i16", "i8"):  # resolve `auto` algorithm choices to the recipe defaults
        legal = generic.legal_keys(fmt, tgt)
        for k in ("unpack", "correction", "scales", "accum"):
            if c[k] == "auto":
                c[k] = legal[k][0]
    for hook in hooks.RESOLVE_HOOKS:
        hook(c)
    c["act_format"] = FORMATS[fmt].act
    c.setdefault("kernel", f"{fmt}_{op}_{tgt}")
    return c


def validate_space(spec, space):
    """Reject unknown tune keys and values that are legal in no combination (typos such
    as rows=3). Returns the number of legal combinations."""
    tunable = ("target",) + tuple(SCHEDULE)
    for k in space:
        if k not in tunable:
            raise SpecError(f"tune key {k!r}: expected one of {sorted(tunable)}")
    n, seen = 0, {k: set() for k in space}
    for ov, _ in iter_space(spec, space):
        n += 1
        for k, v in ov.items():
            seen[k].add(v)
    for k, vs in space.items():
        dead = [v for v in vs if v not in seen[k]]
        if dead:
            raise SpecError(f"tune {k}: {dead} are not legal in any combination of this space")
    return n


def iter_space(spec, space):
    """Yield (overrides, config) for every legal combination in a tune space."""
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        ov = dict(zip(keys, combo))
        try:
            yield ov, resolve(spec, ov)
        except SpecError:
            continue


def legal_configs(op=None, weights=None, target=None, prefetch=(0, 8), enumerated=False):
    """Every legal codegen configuration (optionally filtered), deduplicated on CODEGEN_KEYS.
    Layouts with a hooks.ENUMERATORS entry (layout=composed) are open-ended: they are left
    out of the product, and `enumerated=True` adds their covering sets."""
    keys = tuple(k for k in SCHEDULE if k not in RUNTIME_KEYS)
    pf = keys.index("prefetch")
    seen = set()
    enum = hooks.ENUMERATORS

    def emit(c):
        key = tuple(c[k] for k in CODEGEN_KEYS)
        if key not in seen:  # `auto` and its resolved value are the same config
            seen.add(key)
            return True
        return False

    for (o, f), targets in TARGETS.items():
        if (op and o != op) or (weights and f != weights):
            continue
        for t in targets:
            if target and t != target:
                continue
            allowed = [SCHEDULE[k](o, f, t) for k in keys]
            allowed[pf] = tuple(v for v in allowed[pf] if prefetch is None or v in prefetch)
            for i, k in enumerate(keys):
                if k == "layout":
                    allowed[i] = tuple(v for v in allowed[i] if v not in enum)
                elif k in hooks.ENUM_KEYS:
                    allowed[i] = (DEFAULTS[k],)
            for combo in itertools.product(*allowed):
                try:
                    c = resolve({"op": o, "weights": f, "target": t, **dict(zip(keys, combo))})
                except SpecError:
                    continue
                if emit(c):
                    yield c
            for lay, fn in enum.items() if enumerated else ():
                if lay not in SCHEDULE["layout"](o, f, t):
                    continue
                for ov in fn(o, f, t):
                    if prefetch is not None and ov.get("prefetch", 0) not in prefetch:
                        continue
                    try:
                        c = resolve({"op": o, "weights": f, "target": t, **ov})
                    except SpecError:
                        continue
                    if emit(c):
                        yield c
