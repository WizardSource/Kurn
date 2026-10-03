"""`target cuda` specs: keys, legal values, defaults and validation.

A GPU spec uses the same `key value` / `tune` file syntax as CPU specs:

    kernel   q4_0_gemv_cuda
    op       gemv            # gemv (decode; `cols` activation columns per pass) | gemm (int8 tensor cores)
    weights  q4_0
    target   cuda
    arch     sm_80           # compile target; sm_80 code also runs on sm_86/89/90/100/120
    layout   split           # native: ggml blocks as stored | split: 16-byte-aligned quant plane + scale plane
    tpr      32              # threads per output row
    rpb      4               # rows per CUDA block
    sub      2               # lanes per unit (load width = unit bytes / sub)
    unroll   2               # units in flight per lane
    cols     1               # activation columns per pass
    tune     tpr=16,32 rpb=2,4,8 sub=1,2,4 unroll=1,2,4

Problem keys (`n`, `k`, `m`) give the shape a spec is verified or benchmarked on; they do not
change the generated code.
"""

import itertools
import random

from ..spec import SpecError
from ..spec import parse as _parse

# Per weight format: values per unit of GEMV work (`unit`), quant bytes per unit, the ggml block
# (values, bytes), activation format, scale-plane bytes per block in the split layout, and whether
# the int8 tensor-core GEMM supports it.
FORMATS = {
    "q8_0": dict(unit=32, ub=32, block=32, nbytes=34, act="q8_0", sb=2, gemm=True, doc="8-bit, fp16 scale per 32"),
    "q4_0": dict(unit=32, ub=16, block=32, nbytes=18, act="q8_0", sb=2, gemm=True, doc="4-bit, value = d*(q-8)"),
    "iq4_nl": dict(unit=32, ub=16, block=32, nbytes=18, act="q8_0", sb=2, gemm=True, doc="4-bit non-linear codebook"),
    "q4_K": dict(unit=64, ub=32, block=256, nbytes=144, act="q8_K", sb=16, gemm=False, doc="4-bit K-quant (6-bit sub-scales and mins)"),
    "q2_0": dict(unit=64, ub=16, block=64, nbytes=18, act="q8_0", sb=2, gemm=False, doc="2-bit / ternary Bonsai, value = d*(q-1)"),
    "tq2_0": dict(unit=128, ub=32, block=256, nbytes=66, act="q8_K", sb=2, gemm=False, doc="ternary (BitNet b1.58 TQ2_0)"),
    "q1_0": dict(unit=128, ub=16, block=128, nbytes=18, act="q8_0", sb=2, gemm=False, doc="1-bit Bonsai, value = d*(+-1)"),
    "e8p": dict(unit=32, ub=8, block=256, nbytes=66, act="q8_K", sb=2, gemm=False, doc="E8 lattice codebook, 2.06 bpw"),
}
ACT = {"q8_0": dict(block=32, nbytes=34), "q8_K": dict(block=256, nbytes=292)}
ARCHS = ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120")
OPS = ("gemv", "gemm")

SMEM_LIMIT = 48 * 1024  # static shared memory per block


def _fmt(c):
    return FORMATS[c["weights"]]


GEMV_KEYS = {
    "layout": lambda c: ("native", "split"),
    "tpr": lambda c: (4, 8, 16, 32, 64, 128),
    "rpb": lambda c: (1, 2, 4, 8, 16, 32),
    "sub": lambda c: (1, 2, 4),
    "unroll": lambda c: (1, 2, 4),
    "cols": lambda c: (1, 2, 4, 8),
    "minb": lambda c: (0, 1, 2, 4),
    "mins": lambda c: ("bsums", "dp4a") if c["weights"] == "q4_K" else ("none",),
    "unpack": lambda c: ("bits", "lut") if c["weights"] == "q1_0" else ("none",),
}
GEMM_KEYS = {
    "layout": lambda c: ("native", "split"),
    "bm": lambda c: (32, 64, 128),
    "bn": lambda c: (8, 16, 32, 64, 128),
    "wm": lambda c: (1, 2, 4),
    "wn": lambda c: (1, 2, 4),
    "bkb": lambda c: (1, 2, 4),
    "pipe": lambda c: ("sync", "reg2", "async2", "async3"),
    "pad": lambda c: (0, 16),
    "minb": lambda c: (0, 1, 2, 4),
}
DEFAULTS = {
    "gemv": {"layout": "native", "tpr": 32, "rpb": 4, "sub": 2, "unroll": 2, "cols": 1, "minb": 0, "mins": "bsums",
             "unpack": "bits"},
    "gemm": {"layout": "native", "bm": 64, "bn": 32, "wm": 2, "wn": 2, "bkb": 2, "pipe": "reg2", "pad": 16, "minb": 0},
}  # fmt: skip
PROBLEM = {"n": 4096, "k": 4096, "m": 1}
COMMON = ("kernel", "op", "weights", "target", "arch")


def keys_for(op):
    return GEMV_KEYS if op == "gemv" else GEMM_KEYS


def codegen_keys(op):
    return ("op", "weights", "arch") + tuple(keys_for(op))


def gemm_smem(c):
    q = 32 if c["weights"] == "q8_0" else 16
    stages = {"sync": 1, "reg2": 1, "async2": 2, "async3": 3}[c["pipe"]]
    wrow, xrow = c["bkb"] * q + c["pad"], c["bkb"] * 32 + c["pad"]
    return stages * (c["bm"] * wrow + c["bm"] * c["bkb"] * 4 + c["bn"] * xrow + c["bn"] * c["bkb"] * 4)


def threads(c):
    return c["tpr"] * c["rpb"] if c["op"] == "gemv" else c["wm"] * c["wn"] * 32


INVALID = [
    (lambda c: c["op"] == "gemv" and not 32 <= c["tpr"] * c["rpb"] <= 1024, "tpr * rpb (block size) must be 32..1024"),
    (lambda c: c["op"] == "gemv" and c["mins"] == "bsums" and c["sub"] > 2, "mins=bsums needs sub <= 2 (16-value slices)"),
    (lambda c: c["op"] == "gemm" and (c["bm"] // c["wm"]) % 16, "bm / wm must be a multiple of 16 (m16n8k32 tiles)"),
    (lambda c: c["op"] == "gemm" and (c["bn"] // c["wn"]) % 8, "bn / wn must be a multiple of 8 (m16n8k32 tiles)"),
    (lambda c: c["op"] == "gemm" and c["bm"] < c["wm"] * 16, "bm must be >= 16 * wm"),
    (lambda c: c["op"] == "gemm" and c["bn"] < c["wn"] * 8, "bn must be >= 8 * wn"),
    (lambda c: c["op"] == "gemm" and (c["bm"] // c["wm"] // 16) * (c["bn"] // c["wn"] // 8) > 32,
     "warp tile has more than 32 mma tiles (128 accumulators per thread)"),
    (lambda c: c["op"] == "gemm" and c["pipe"].startswith("async") and c["layout"] != "split",
     "pipe=async* needs layout split (16-byte cp.async from aligned planes)"),
    (lambda c: c["op"] == "gemm" and gemm_smem(c) > SMEM_LIMIT, f"shared memory exceeds {SMEM_LIMIT} bytes"),
]  # fmt: skip


def parse(text):
    return _parse(text)


def load(path):
    with open(path) as fh:
        return parse(fh.read())


def resolve(spec, overrides=None):
    """Validate a cuda spec (plus overrides) and fill defaults. Returns the config dict."""
    c = {**spec, **(overrides or {})}
    if c.get("target", "cuda") != "cuda":
        raise SpecError(f"target {c['target']!r}: kurn.gpu handles target cuda only")
    c["target"] = "cuda"
    for req in ("op", "weights"):
        if req not in c:
            raise SpecError(f"missing required key {req!r}")
    if c["op"] not in OPS:
        raise SpecError(f"op {c['op']!r}: expected one of {list(OPS)} for target cuda")
    if c["weights"] not in FORMATS:
        raise SpecError(f"weights {c['weights']!r}: expected one of {list(FORMATS)} for target cuda")
    if c["op"] == "gemm" and not FORMATS[c["weights"]]["gemm"]:
        raise SpecError(f"op gemm supports {[f for f, v in FORMATS.items() if v['gemm']]} (got {c['weights']})")
    keys = keys_for(c["op"])
    known = COMMON + tuple(keys) + tuple(PROBLEM)
    for k in c:
        if k not in known:
            raise SpecError(f"unknown key {k!r} for op {c['op']} target cuda: expected one of {sorted(known)}")
    c.setdefault("arch", "sm_80")
    if c["arch"] not in ARCHS:
        raise SpecError(f"arch {c['arch']!r}: expected one of {list(ARCHS)}")
    for k, fn in keys.items():
        allowed = fn(c)
        if k not in c:
            d = DEFAULTS[c["op"]][k]
            c[k] = d if d in allowed else allowed[0]
        if c[k] not in allowed:
            raise SpecError(f"{k}={c[k]!r} not allowed for {c['op']}/{c['weights']}/cuda: expected one of {list(allowed)}")
    if (
        c["op"] == "gemm"
        and c["layout"] == "native"
        and c["pipe"].startswith("async")
        and "pipe" not in spec
        and "pipe" not in (overrides or {})
    ):
        c["pipe"] = "reg2"
    for bad, msg in INVALID:
        if bad(c):
            raise SpecError(msg)
    for k, v in PROBLEM.items():
        c.setdefault(k, v)
    c.setdefault("kernel", f"{c['weights']}_{c['op']}_cuda")
    return c


def config_key(c):
    return tuple(c[k] for k in codegen_keys(c["op"]))


def label(c):
    return " ".join(f"{k}={c[k]}" for k in codegen_keys(c["op"]) if k not in ("op", "weights"))


def legal_configs(op, weights, arch="sm_80"):
    """Every legal codegen configuration for (op, weights)."""
    keys = keys_for(op)
    base = {"op": op, "weights": weights, "target": "cuda", "arch": arch}
    probe = dict(base)
    names = list(keys)
    for combo in itertools.product(*(keys[k](probe) for k in names)):
        try:
            yield resolve({**base, **dict(zip(names, combo))})
        except SpecError:
            continue


def covering_configs(op, weights, arch="sm_80", extra=24, seed=0):
    """A covering set for verification: the defaults, every legal value of every key varied
    one at a time from the defaults (or from the nearest legal neighbour), plus `extra` random
    legal configurations. Deduplicated on the codegen keys."""
    base = {"op": op, "weights": weights, "target": "cuda", "arch": arch}
    keys = keys_for(op)
    out, seen = [], set()

    def add(c):
        if config_key(c) not in seen:
            seen.add(config_key(c))
            out.append(c)

    d = resolve(base)
    add(d)
    rng = random.Random(seed)
    for k, fn in keys.items():
        for v in fn(d):
            try:
                add(resolve(base, {k: v}))
                continue
            except SpecError:
                pass
            for _ in range(200):  # find a legal neighbour that has k=v
                ov = {k2: rng.choice(fn2(d)) for k2, fn2 in keys.items() if k2 != k}
                try:
                    add(resolve(base, {**ov, k: v}))
                    break
                except SpecError:
                    continue
    names = list(keys)
    tries = 0
    target = len(out) + extra
    while len(out) < target and tries < 50 * extra:
        tries += 1
        try:
            add(resolve(base, {k: rng.choice(keys[k](d)) for k in names}))
        except SpecError:
            pass
    return out


def validate_space(spec, space):
    n = 0
    for _ov, _c in iter_space(spec, space):
        n += 1
    if not n:
        raise SpecError("tune space has no legal configuration")
    return n


def iter_space(spec, space):
    names = list(space)
    for combo in itertools.product(*(space[k] for k in names)):
        ov = dict(zip(names, combo))
        try:
            yield ov, resolve(spec, ov)
        except SpecError:
            continue
