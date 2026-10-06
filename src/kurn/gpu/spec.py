"""`target cuda` specs: keys, legal values, defaults and validation.

A GPU spec uses the same `key value` / `tune` file syntax as CPU specs:

    kernel   q4_0_gemv_cuda
    op       gemv            # gemv (decode; `cols` activation columns per pass) | gemm (int8 tensor cores)
    weights  q4_0
    target   cuda
    arch     sm_80           # compile target; sm_80 code also runs on sm_86/89/90/100/120
    layout   split           # native: ggml blocks as stored | split: 16-byte-aligned quant plane + scale plane
    xlayout  blocks          # activations: ggml q8_0 blocks | split: aligned int8 plane + scale plane (q8_0-activation formats)
    tpr      32              # threads per output row
    rpb      4               # rows per CUDA block
    sub      2               # lanes per unit (load width = unit bytes / sub)
    unroll   2               # units in flight per lane
    cols     1               # activation columns per pass
    tune     tpr=16,32 rpb=2,4,8 sub=1,2,4 unroll=1,2,4

Problem keys (`n`, `k`, `m`) give the shape a spec is verified or benchmarked on; they do not
change the generated code.
"""

import functools
import itertools
import json
import random
from importlib import resources

from ..spec import SpecError
from ..spec import parse as _parse

# Per weight format: values per unit of GEMV work (`unit`), quant bytes per unit, the ggml block
# (values, bytes), activation format, scale-plane bytes per block in the split layout, and whether
# the int8 tensor-core GEMM supports it.
FORMATS = {
    "q8_0": dict(unit=32, ub=32, block=32, nbytes=34, act="q8_0", sb=2, gemm=True, doc="8-bit, fp16 scale per 32"),
    "q4_0": dict(unit=32, ub=16, block=32, nbytes=18, act="q8_0", sb=2, gemm=True, doc="4-bit, value = d*(q-8)"),
    "iq4_nl": dict(unit=32, ub=16, block=32, nbytes=18, act="q8_0", sb=2, gemm=True, doc="4-bit non-linear codebook"),
    "q4_K": dict(unit=64, ub=32, block=256, nbytes=144, act="q8_K", sb=16, gemm=True, doc="4-bit K-quant (6-bit sub-scales and mins)"),
    "q2_0": dict(unit=64, ub=16, block=64, nbytes=18, act="q8_0", sb=2, gemm=True, doc="2-bit / ternary Bonsai, value = d*(q-1)"),
    "tq2_0": dict(unit=128, ub=32, block=256, nbytes=66, act="q8_K", sb=2, gemm=True, doc="ternary (BitNet b1.58 TQ2_0)"),
    "q1_0": dict(unit=128, ub=16, block=128, nbytes=18, act="q8_0", sb=2, gemm=True, doc="1-bit Bonsai, value = d*(+-1)"),
    "e8p": dict(unit=32, ub=8, block=256, nbytes=66, act="q8_K", sb=2, gemm=True, doc="E8 lattice codebook, 2.06 bpw"),
    "mxfp4": dict(unit=32, ub=16, block=32, nbytes=17, act="q8_0", sb=1, gemm=True, doc="OCP MXFP4: E2M1 codes, E8M0 scale per 32"),
    "nvfp4": dict(
        unit=32,
        ub=16,
        block=64,
        nbytes=36,
        act="q8_0",
        sb=4,
        gemm=True,
        doc="NVFP4: E2M1 codes, UE4M3 scale per 16 (per-tensor scale applied by the caller)",
    ),
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
    "xlayout": lambda c: ("blocks", "split") if _fmt(c)["act"] == "q8_0" else ("blocks",),
}


def _bk(c):
    from .mma import ENGINE

    return tuple(b for b in (64, 128, 256) if b % ENGINE[c["weights"]]["kt"] == 0)


GEMM_KEYS = {
    "bm": lambda c: (16, 32, 64, 128, 256),
    "bn": lambda c: (8, 16, 32, 64, 128, 256),
    "wm": lambda c: (1, 2, 4, 8),
    "wn": lambda c: (1, 2, 4, 8),
    "bk": _bk,
    "stages": lambda c: (2, 3, 4, 5),
    "splitk": lambda c: (0, 1, 2, 4, 8, 16),
    "xin": lambda c: ("f32", "f16"),
    "minb": lambda c: (1, 2),
}
DEFAULTS = {
    "gemv": {"layout": "native", "tpr": 32, "rpb": 4, "sub": 2, "unroll": 2, "cols": 1, "minb": 0, "mins": "bsums",
             "unpack": "bits", "xlayout": "blocks"},
    "gemm": {"bm": 64, "bn": 8, "wm": 4, "wn": 1, "bk": 256, "stages": 4, "splitk": 0, "xin": "f32", "minb": 1},
}  # fmt: skip
# GEMV defaults per format: the split layout (16-byte aligned quant plane) with `sub` chosen so every lane issues 16-byte
# weight loads, and `unroll` units in flight per lane (2 where registers are tight). From the A100 run: tuning Q4_0 this
# way gave 1.30x over the old native/sub=2/unroll=2 default. xlayout split (aligned activations, contributed by the user)
# is the default where it exists: on the default kernels it cuts global loads 3-5x and hot-loop instructions 14-40% (SASS).
GEMV_FORMAT_DEFAULTS = {
    "q8_0": {"layout": "split", "sub": 2, "unroll": 4, "xlayout": "split"},
    "q4_0": {"layout": "split", "sub": 1, "unroll": 4, "xlayout": "split"},
    "iq4_nl": {"layout": "split", "sub": 1, "unroll": 4, "xlayout": "split"},
    "q4_K": {"layout": "split", "sub": 2, "unroll": 2},
    "q2_0": {"layout": "split", "sub": 1, "unroll": 4, "xlayout": "split"},
    "tq2_0": {"layout": "split", "sub": 2, "unroll": 2},
    "q1_0": {"layout": "split", "sub": 1, "unroll": 4, "xlayout": "split"},
    "e8p": {"layout": "split", "sub": 1, "unroll": 4},
    "mxfp4": {"layout": "split", "sub": 1, "unroll": 4, "xlayout": "split"},
    "nvfp4": {"layout": "split", "sub": 1, "unroll": 4, "xlayout": "split"},
}
PROBLEM = {"n": 4096, "k": 4096, "m": 1}
COMMON = ("kernel", "op", "weights", "target", "arch")


def keys_for(op):
    return GEMV_KEYS if op == "gemv" else GEMM_KEYS


def codegen_keys(op):
    return ("op", "weights", "arch") + tuple(keys_for(op))


def gemm_smem(c):
    from .mma import smem_bytes

    return smem_bytes(c)


def gemm_smem_limit(c):
    from .mma import SMEM_MAX

    return SMEM_MAX[c["arch"]]


REG_LIMIT = 224  # est_regs beyond which ptxas spills (10% margin to the launch-bounds cap); the tuner also drops spills


def _cols_unroll_max(c):
    return 4 if c["weights"] == "tq2_0" else 8 if c["weights"] in ("q4_K", "e8p", "q8_0") else 16


def _reg_budget(c):
    margin = 0.75 if c["minb"] > 1 else 0.9
    return min(REG_LIMIT - (20 if c["weights"] == "q4_K" else 0), int(margin * _reg_cap(c)))


def _est_regs(c):
    from .mma import est_regs

    return est_regs(c)


def _reg_cap(c):
    from .mma import reg_cap

    return reg_cap(c)


def _deep_staged_tile(c):
    from .mma import deep_staged_tile

    return deep_staged_tile(c)


def threads(c):
    return c["tpr"] * c["rpb"] if c["op"] == "gemv" else c["wm"] * c["wn"] * 32


def gemv_threads_key(weights, xlayout, sub, cols):
    return f"{weights} {xlayout} sub={sub} cols={cols}"


@functools.cache
def _gemv_threads_table():
    return json.loads((resources.files("kurn.gpu") / "data" / "gemv_threads.json").read_text())["limits"]


def gemv_threads_max(c):
    """Largest tpr * rpb * max(1, minb) at which this GEMV compiles without spills (ptxas-measured over every tpr,
    layout, mins and unpack on sm_80/90/100 by tools/gemv_threads.py); 0 if it spills at any block size."""
    row = _gemv_threads_table().get(gemv_threads_key(c["weights"], c["xlayout"], c["sub"], c["cols"]), {})
    return row.get(str(c["unroll"]), 0)


GEMV_SPILL_RULES = [
    (lambda c: c["op"] == "gemv" and c["tpr"] * c["rpb"] * max(1, c["minb"])
     > (256 if c["cols"] >= 8 or c["cols"] * c["unroll"] >= 16 or (c["cols"] >= 4 and c["weights"] == "tq2_0")
        else 512 if c["cols"] > 1 or (c["unroll"] > 2 and c["weights"] in ("q4_K", "e8p", "tq2_0", "q8_0")) else 1024),
     "minb x block size leaves too few registers per thread (would spill)"),
    (lambda c: c["op"] == "gemv" and c["cols"] * c["unroll"] > _cols_unroll_max(c),
     "cols * unroll too large (register spills; use op gemm for batches above 8)"),
    (lambda c: c["op"] == "gemv" and gemv_threads_max(c) == 0,
     "cols * unroll too large for this format, xlayout and sub (ptxas spills at any block size; use op gemm for batches"
     " above 8)"),
    (lambda c: c["op"] == "gemv" and c["tpr"] * c["rpb"] * max(1, c["minb"]) > gemv_threads_max(c),
     "minb x block size leaves too few registers per thread for this format, xlayout, sub, cols and unroll (ptxas"
     " spills; limits in kurn/gpu/data/gemv_threads.json)"),
]  # fmt: skip

INVALID = [
    (lambda c: c["op"] == "gemv" and not 32 <= c["tpr"] * c["rpb"] <= 1024, "tpr * rpb (block size) must be 32..1024"),
    (lambda c: c["op"] == "gemv" and c["mins"] == "bsums" and c["sub"] > 2, "mins=bsums needs sub <= 2 (16-value slices)"),
    (lambda c: c["op"] == "gemv" and c["weights"] == "q4_K" and c["cols"] >= 4 and c["sub"] < 2,
     "q4_K with 4+ columns needs sub >= 2 (register pressure)"),
    *GEMV_SPILL_RULES,
    (lambda c: c["op"] == "gemm" and (c["bm"] % (16 * c["wm"]) or c["bn"] % (8 * c["wn"])),
     "warp tile (bm/wm x bn/wn) must be a multiple of 16 x 8 (m16n8k16 tiles)"),
    (lambda c: c["op"] == "gemm" and not 32 <= c["wm"] * c["wn"] * 32 <= 512, "wm * wn warps must give 32..512 threads"),
    (lambda c: c["op"] == "gemm" and c["bm"] // c["wm"] > 64, "warp tile at most 64 rows (bm / wm <= 64)"),
    (lambda c: c["op"] == "gemm" and (c["bm"] // c["wm"] // 16) * (c["bn"] // c["wn"] // 8) > 32,
     "warp tile has more than 32 mma tiles (128 accumulators per thread)"),
    (lambda c: c["op"] == "gemm" and c["xin"] == "f32" and c["bn"] * (c["bk"] // 32) > c["wm"] * c["wn"] * 32,
     "xin=f32 stages one (column, 32-value window) per thread: needs bn * bk/32 <= threads (use xin=f16 for large tiles)"),
    (lambda c: c["op"] == "gemm" and gemm_smem(c) > gemm_smem_limit(c), "shared memory exceeds the arch's per-block limit"),
    (lambda c: c["op"] == "gemm" and _est_regs(c) > _reg_budget(c),
     "tile needs too many registers for its launch bounds (would spill)"),
    (lambda c: c["op"] == "gemm" and _deep_staged_tile(c),
     "64-row warp tile with 2+ n8 tiles and 2+ k-tiles per stage stages 12+ cp.async chunks per thread: too many registers"
     " (ptxas spills; use more warps or a smaller bk)"),
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
    fdef = GEMV_FORMAT_DEFAULTS.get(c["weights"], {}) if c["op"] == "gemv" else {}
    for k, fn in keys.items():
        allowed = fn(c)
        if k not in c:
            d = fdef.get(k, DEFAULTS[c["op"]][k])
            c[k] = d if d in allowed else allowed[0]
        if c[k] not in allowed:
            raise SpecError(f"{k}={c[k]!r} not allowed for {c['op']}/{c['weights']}/cuda: expected one of {list(allowed)}")
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
