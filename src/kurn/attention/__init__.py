"""`op attn`: tiled (FlashAttention-style) CPU attention with online softmax.

A spec names the kernel configuration and, optionally, the problem it is tuned on:

    op        attn
    kv        q8_0           # KV cache format: f16 | bf16 | q8_0
    target    amx_bf16       # tile engine: avx512 (f32 FMA) | avx512_bf16 (vdpbf16ps) | amx_bf16
    dk        128            # head dims (dv defaults to dk; MLA: dk 576 dv 512)
    tile_q    64             # query tokens per tile (x G query heads = rows per tile)
    tile_kv   128            # KV tokens per tile
    split     0              # KV splits per (kv head, query tile), 0 = auto (flash-decoding)
    dec_rows  8              # AVX-512 row engine when n_q * G <= dec_rows (decode), 0 = never
    threads   8
    heads 16  kv_heads 8  nq 4096  nkv 4096          # problem (not codegen)
    tune      tile_q=32,64,128 tile_kv=64,128,256

Generated C implements kurn_attn.h (`kattn`, `kattn_workspace`, `kattn_config`); the
template is data/attn_kernel.c and the harness data/bench_attn.c.
"""

import csv
import hashlib
import itertools
import os
import shlex
import subprocess
import tempfile

from .. import toolchain
from ..spec import SpecError, parse
from ..tune import PROXY_W_PER_CORE, pareto_front

_V4 = frozenset({"avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"})
# target -> (compiler flags, /proc/cpuinfo flags needed, engine id in the template)
TARGETS = {
    "avx512": (("-O3", "-march=x86-64-v4"), _V4, 0),
    "avx512_bf16": (("-O3", "-march=x86-64-v4", "-mavx512bf16"), _V4 | {"avx512_bf16"}, 1),
    "amx_bf16": (("-O3", "-march=x86-64-v4", "-mavx512bf16", "-mamx-tile", "-mamx-bf16"),
                 _V4 | {"avx512_bf16", "amx_tile", "amx_bf16"}, 2),
}  # fmt: skip
KV_FORMATS = {"f16": 0, "bf16": 1, "q8_0": 2}
KV_ROW_BYTES = {"f16": lambda d: 2 * d, "bf16": lambda d: 2 * d, "q8_0": lambda d: d // 32 * 34}
HEAD_DIMS = {64: (64,), 128: (128,), 256: (256,), 576: (512,)}  # dk -> legal dv

# Legal values per key, as a function of the partially resolved config.
SCHEDULE = {
    "kv": lambda c: tuple(KV_FORMATS),
    "dk": lambda c: tuple(HEAD_DIMS),
    "dv": lambda c: HEAD_DIMS[c["dk"]],
    "tile_q": lambda c: (16, 32, 64, 128, 256),
    "tile_kv": lambda c: (64, 128, 256),
    "split": lambda c: (0, 1, 2, 4, 8, 16, 32, 64),
    "dec_rows": lambda c: (0, 4, 8),
    "threads": lambda c: tuple(range(1, 257)),
}
DEFAULTS = {"kv": "f16", "dk": 128, "tile_q": 64, "tile_kv": 128, "split": 0, "dec_rows": 8,
            "threads": min(8, os.cpu_count() or 8)}  # fmt: skip
CODEGEN_KEYS = ("kv", "target", "dk", "dv", "tile_q", "tile_kv", "split", "dec_rows")
# Problem keys: the shape a spec is verified / tuned on (do not change the generated C).
PROBLEM = {"heads": 16, "kv_heads": 8, "nq": 1, "nkv": 4096, "pos0": -1, "causal": 1, "mask": 0, "mla": 0}
KNOWN_KEYS = ("kernel", "op", "target") + tuple(SCHEDULE) + tuple(PROBLEM)
# Accepted max |out - ref| / max |ref| against the float64 reference on the same (dequantized)
# K/V: the f32 engines differ only by summation order and the exp2 polynomial; the bf16 engines
# round q, k (from f16) and the probabilities to bf16 (8-bit mantissa).
TOL = {"avx512": 1e-4, "avx512_bf16": 1e-2, "amx_bf16": 1e-2}


def resolve(spec, overrides=None):
    """Validate a spec dict (plus overrides) and fill defaults. Returns the config dict."""
    c = {**spec, **(overrides or {})}
    if c.get("op", "attn") != "attn":
        raise SpecError(f"op {c['op']!r}: kurn.attention only handles `op attn`")
    c["op"] = "attn"
    if "target" not in c:
        raise SpecError("missing required key 'target'")
    if c["target"] not in TARGETS:
        raise SpecError(f"target {c['target']!r}: expected one of {list(TARGETS)}")
    for k in c:
        if k not in KNOWN_KEYS:
            raise SpecError(f"unknown key {k!r}: expected one of {sorted(KNOWN_KEYS)}")
    for k, legal in SCHEDULE.items():
        allowed = legal(c)
        if k not in c:
            c[k] = DEFAULTS[k] if DEFAULTS.get(k) in allowed else allowed[0]
        if c[k] not in allowed:
            shown = "an integer in 1..256" if k == "threads" else f"one of {list(allowed)}"
            raise SpecError(f"{k}={c[k]!r} not allowed for attn/{c['target']}: expected {shown}")
    for k, v in PROBLEM.items():
        c.setdefault(k, v)
        if not isinstance(c[k], int):
            raise SpecError(f"{k}={c[k]!r}: expected an integer")
    if c["heads"] % c["kv_heads"]:
        raise SpecError(f"heads={c['heads']} must be a multiple of kv_heads={c['kv_heads']}")
    if c["mla"] and c["dv"] > c["dk"]:
        raise SpecError("mla needs dv <= dk (v is the first dv values of each k row)")
    c.setdefault("kernel", f"attn_{c['kv']}_d{c['dk']}_{c['target']}")
    return c


def load(path):
    with open(path) as fh:
        return parse(fh.read())


def _data(name):
    return toolchain.data_path(name)


def generate(c):
    """Resolved config -> C source implementing kurn_attn.h."""
    with open(_data("attn_kernel.c")) as fh:
        body = fh.read()
    with open(_data("kurn_attn.h"), "rb") as fh:
        hsha = hashlib.sha1(fh.read()).hexdigest()[:12]
    head = [f"/* kurn attention kernel: {' '.join(f'{k}={c[k]}' for k in CODEGEN_KEYS)} (kurn_attn.h {hsha}) */"]
    defs = {"KA_DK": c["dk"], "KA_DV": c["dv"], "KA_KV": KV_FORMATS[c["kv"]], "KA_ENGINE": TARGETS[c["target"]][2],
            "KA_TQ": c["tile_q"], "KA_TK": c["tile_kv"], "KA_SPLIT": c["split"], "KA_DEC_ROWS": c["dec_rows"]}  # fmt: skip
    head += [f"#define {k} {v}" for k, v in defs.items()]
    return "\n".join(head) + "\n" + body


def runnable(target):
    missing = sorted(TARGETS[target][1] - toolchain.cpu_flags())
    return (not missing and toolchain.host_arch() == "x86_64"), ", ".join(missing)


def _compile(src, flags, stem, out_dir=None, shared=True, extra=()):
    cc = toolchain.host_cc()
    args_sig = shlex.join([*cc, *flags, *extra])
    h = hashlib.sha1((src + "\0" + args_sig + "\0" + str(shared)).encode()).hexdigest()[:12]
    d = out_dir or os.path.join(toolchain.cache_dir(), "attn")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{stem}_{h}")
    out = base + (".so" if shared else "")
    if os.path.exists(out):
        return out
    with open(base + ".c", "w") as fh:
        fh.write(src)
    args = [*cc, *flags, *extra, "-I", os.path.dirname(_data("kurn_attn.h"))]
    args += ["-shared", "-fPIC", base + ".c"] if shared else [base + ".c", "-lpthread", "-ldl", "-lm"]
    tmp = f"{out}.tmp{os.getpid()}"
    r = subprocess.run([*args, "-o", tmp], capture_output=True, text=True)
    if r.returncode:
        raise toolchain.BuildError(f"C compile failed:\n$ {shlex.join(args)}\n{r.stderr[:4000]}")
    os.replace(tmp, out)
    return out


def build(c, out_dir=None, extra_flags=()):
    """Generate and compile a resolved config; returns the shared library path."""
    return _compile(generate(c), TARGETS[c["target"]][0], c["kernel"], out_dir, extra=tuple(extra_flags))


def build_harness():
    with open(_data("bench_attn.c")) as fh:
        src = fh.read()
    with open(_data("kurn_attn.h")) as fh:
        src = f"/* kurn_attn.h {hashlib.sha1(fh.read().encode()).hexdigest()[:12]} */\n" + src
    return _compile(src, ("-O3", "-march=native"), "bench_attn", shared=False)


CSV_COLUMNS = ("impl,kernel,regime,threads,n_q,n_kv,heads,kv_heads,calls,wall_s,cpu_s,us_per_call,GFLOPs,GBps,"
               "proxy_uJ_per_call,relerr,check,drift_s,dv,mla").split(",")  # fmt: skip


class HarnessError(Exception):
    pass


def problem_args(c):
    a = ["--nq", c["nq"], "--nkv", c["nkv"], "--heads", c["heads"], "--kv-heads", c["kv_heads"], "--causal", c["causal"]]
    if c["pos0"] >= 0:
        a += ["--pos0", c["pos0"]]
    if c["mask"]:
        a += ["--mask"]
    if c["mla"]:
        a += ["--mla"]
    return [str(x) for x in a]


def bench(so, c, regime="hot", secs=1.0, extra=(), tol=None, timeout=1800):
    """Run one kernel on the problem in `c`. Returns the harness CSV row as a dict (+ `us`, `cpu_us`)."""
    ok, missing = runnable(c["target"])
    if not ok:
        raise HarnessError(f"cannot run {c['target']} here: host CPU lacks {missing}")
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    os.remove(tmp)
    cmd = [build_harness(), "--impl", so, "--threads", str(c["threads"]), "--regime", regime, "--secs", str(secs),
           "--tol", str(tol if tol is not None else TOL[c["target"]]), "--csv", tmp, *problem_args(c), *extra]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if not os.path.exists(tmp):
        raise HarnessError(f"harness failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[:2000]}")
    with open(tmp) as fh:
        row = dict(zip(CSV_COLUMNS, fh.read().strip().split(",")))
    os.remove(tmp)
    calls = float(row["calls"])
    row["us"] = float(row["us_per_call"])
    row["cpu_us"] = float(row["cpu_s"]) / calls * 1e6
    for k in ("relerr", "GBps", "GFLOPs"):
        row[k] = float(row[k])
    return row


# Awkward shapes for a correctness check: query-tile tails, GQA ratios 1/2/4, causal and
# explicit-mask paths, decode with auto and forced KV splits, tails of the KV tile.
CHECK_SHAPES = (
    {"nq": 1, "nkv": 1000, "heads": 8, "kv_heads": 2},
    {"nq": 3, "nkv": 777, "heads": 4, "kv_heads": 4},
    {"nq": 1, "nkv": 4100, "heads": 16, "kv_heads": 8},
    {"nq": 100, "nkv": 100, "heads": 4, "kv_heads": 2},
    {"nq": 37, "nkv": 300, "heads": 6, "kv_heads": 3},
    {"nq": 70, "nkv": 333, "heads": 8, "kv_heads": 2, "mask": 1},
    {"nq": 64, "nkv": 200, "heads": 2, "kv_heads": 1, "causal": 0},
    {"nq": 129, "nkv": 1500, "heads": 4, "kv_heads": 1},
)


def check(so, c, shapes=CHECK_SHAPES, log=None):
    """Numerical check on awkward shapes. Returns the worst row (check == "FAIL" if any failed)."""
    worst = None
    for sh in shapes:
        cc = {**c, **PROBLEM, "pos0": -1, **sh, "threads": min(3, c["threads"])}
        if c["mla"]:
            cc.update(mla=1, heads=sh["heads"], kv_heads=1)
        row = bench(so, cc, "hot", 0, ("--check-toks", "40"))
        if log:
            log(f"  {sh}: relerr {row['relerr']:.2e} {row['check']}")
        if worst is None or row["check"] == "FAIL" or row["relerr"] > worst["relerr"]:
            if not (worst and worst["check"] == "FAIL" and row["check"] != "FAIL"):
                worst = row
    return worst


def energy_uj(row, static_w=0.0):
    return row["cpu_us"] * PROXY_W_PER_CORE + row["us"] * static_w


def tune(spec, space, regime="hot", objective="energy", static_w=0.0, secs=1.0, out=None, log=print):
    """Sweep `space` on top of `spec` (problem from the spec). Returns (ranked results, pareto front)."""
    key = {"energy": "energy_uJ", "speed": "us", "edp": "edp"}[objective]
    keys, results, seen = list(space), [], set()
    for combo in itertools.product(*(space[k] for k in keys)):
        ov = dict(zip(keys, combo))
        try:
            c = resolve(spec, ov)
        except SpecError as e:
            log(f"skip {ov}: {e}")
            continue
        sig = tuple(c[k] for k in CODEGEN_KEYS + ("threads",))
        if sig in seen:
            continue
        seen.add(sig)
        try:
            row = bench(build(c), c, regime, secs)
        except (toolchain.BuildError, HarnessError) as e:
            log(f"FAIL {ov}: {str(e).splitlines()[0]}")
            continue
        if row["check"] == "FAIL":
            log(f"FAIL {ov}: relerr {row['relerr']:.2e}")
            continue
        e = energy_uj(row, static_w)
        results.append({**ov, "us": row["us"], "cpu_us": row["cpu_us"], "energy_uJ": e, "edp": e * row["us"],
                        "GFLOPs": row["GFLOPs"], "GBps": row["GBps"], "relerr": row["relerr"]})  # fmt: skip
        log(f"{ov}  {row['us']:10.1f} us  {e:10.1f} uJ  {row['GFLOPs']:8.1f} GFLOP/s  {row['GBps']:7.1f} GB/s  relerr {row['relerr']:.1e}")
    results.sort(key=lambda r: r[key])
    if out and results:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    return results, pareto_front(results)


def legal_configs(kv=None, target=None, dk=(64, 128)):
    """Codegen configurations of the closed space (for tests; tile/split values at their defaults
    unless they change the engine structure)."""
    for t, f, d, tq, tk in itertools.product(TARGETS, KV_FORMATS, dk, (16, 64), (64, 128)):
        if (kv and f != kv) or (target and t != target):
            continue
        yield resolve({"op": "attn", "target": t, "kv": f, "dk": d, "tile_q": tq, "tile_kv": tk})
