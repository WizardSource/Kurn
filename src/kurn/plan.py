"""Plan cache (TensorRT-style): tune once per (machine, op, format, shape, regime), store
the winner, reuse it at model load.

    kurn plan build MODEL.gguf [--threads T] [--ops gemv,verify] [--regime cold|hot]
    kurn plan show [--model MODEL.gguf] [--json]
    kurn plan lookup OP FORMAT K N [--regime cold8]

A plan file lives at <cache>/plans/<fingerprint id>.json. The fingerprint is everything
that can change which kernel wins: CPU model, ISA flags, cache sizes, core count, the C
compiler, and the kurn version (a new lowering invalidates old winners).

Entries are keyed "op/format/K<K>xN<N>/<regime><threads>" (cols are part of the config).
Building an entry is multi-fidelity: a staged search (tune.search) on the per-core slice
(N / threads rows, hot) ranks the space cheaply; the best `top` configs, plus the default
config as a baseline, are then re-measured interleaved in the target regime (e.g. cold,
8 threads, DRAM streaming), and the winner by the objective is stored.

Other workstreams: `lookup(op, fmt, K, N, regime="cold", threads=None)` returns a resolved
config (or None); `for_model(path)` maps every supported matmul tensor of a GGUF to one.
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time

from . import __version__
from .harness import HarnessError, bench
from .kernels import KERNELS
from .spec import CODEGEN_KEYS, RUNTIME_KEYS, SpecError, resolve
from .toolchain import BuildError, build, cache_dir, cpu_flags, host_cc, run_mode
from .tune import OBJECTIVES, _finite_number, _positive_int, measurement, search, summarize_measurements

PLAN_FLAGS = ("avx2", "fma", "f16c", "avx_vnni", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16",
              "avx512_fp16", "avx512vbmi", "avx512_vbmi2", "amx_tile", "amx_int8", "amx_bf16", "asimd", "asimddp", "sve")


# --------------------------------------------------------------------------- fingerprint
def _cpu_model():
    """Model name plus family/model/stepping (VMs often report a generic model name)."""
    info = {}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if not line.strip():
                    break
                k, _, v = line.partition(":")
                info[k.strip().lower()] = v.strip()
    except OSError:
        pass
    name = info.get("model name") or info.get("cpu model") or info.get("hardware") or platform.processor() or platform.machine()
    ids = "-".join(info[k] for k in ("cpu family", "model", "stepping") if k in info)
    return f"{name} [{ids}]" if ids else name


def _caches():
    out = {}
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        for d in sorted(os.listdir(base)):
            if not d.startswith("index"):
                continue
            info = {}
            for f in ("type", "level", "size"):
                with open(os.path.join(base, d, f)) as fh:
                    info[f] = fh.read().strip()
            if info["type"] == "Instruction":
                continue
            out[f"L{info['level']}{'d' if info['type'] == 'Data' else ''}"] = info["size"]
    except OSError:
        pass
    return out


def _cc_version():
    try:
        r = subprocess.run([*host_cc(), "--version"], capture_output=True, text=True, timeout=30)
        return r.stdout.splitlines()[0].strip() if r.stdout else ""
    except Exception:  # noqa: BLE001  (no compiler is still a valid fingerprint)
        return ""


def fingerprint():
    flags = cpu_flags()
    return {
        "cpu": _cpu_model(),
        "arch": platform.machine(),
        "flags": sorted(f for f in PLAN_FLAGS if f in flags),
        "caches": _caches(),
        "cores": os.cpu_count() or 1,
        "cc": _cc_version(),
        "kurn": __version__,
    }


def fingerprint_id(fp=None):
    fp = fingerprint() if fp is None else fp
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


def plan_path(fp=None):
    d = os.path.join(cache_dir(), "plans")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{fingerprint_id(fp)}.json")


# --------------------------------------------------------------------------- model shapes
def _kurn_format(type_name, op="gemv"):
    for o, f in KERNELS:
        if o == op and f.lower() == type_name.lower():
            return f
    return None


def model_shapes(path):
    """[(tensor name, gguf type, kurn format or None, K, N)] for every 2-D weight tensor.
    token_embd is skipped unless the model has no output.weight (tied embeddings)."""
    import gguf

    r = gguf.GGUFReader(path)
    names = {t.name for t in r.tensors}
    out = []
    for t in r.tensors:
        if len(t.shape) != 2 or not t.name.endswith(".weight"):
            continue
        if t.name == "token_embd.weight" and "output.weight" in names:
            continue
        try:
            tn = t.tensor_type.name
        except Exception:  # noqa: BLE001  (types newer than the gguf package)
            tn = str(t.tensor_type)
        out.append((t.name, tn, _kurn_format(tn), int(t.shape[0]), int(t.shape[1])))
    return out


def default_target(op="gemv", fmt="q8_0"):
    """Fastest preferred *supported native* ISA, restricted to this kernel pair.

    AVX2-only hosts must not silently fall back to scalar for formats with an
    AVX2 lowering. AMX remains opt-in, as in the upstream implementation.
    """
    k = KERNELS.get((op, fmt))
    if k is None:
        raise SpecError(f"unsupported kernel {op}/{fmt}")
    for target in ("avx512_vnni", "avx2_vnni", "avx2", "neon", "scalar"):
        if target in k.targets and run_mode(target)[0] == "native":
            return target
    raise SpecError(f"no native target available for {op}/{fmt}; specify a supported target")


def entry_key(op, fmt, K, N, regime, threads):
    return f"{op}/{fmt}/K{K}xN{N}/{regime}{threads}"


# --------------------------------------------------------------------------- plan file
class Plan:
    def __init__(self, path=None, fp=None):
        self.fp = fingerprint() if fp is None else dict(fp)
        self.path = path or plan_path(self.fp)
        self.data = {"fingerprint": self.fp, "id": fingerprint_id(self.fp), "entries": {}, "models": {}}
        if os.path.exists(self.path):
            try:
                with open(self.path) as fh:
                    loaded = json.load(fh)
            except (OSError, ValueError) as e:
                raise SpecError(f"cannot read plan {self.path}: {e}") from e
            if not isinstance(loaded, dict) or loaded.get("fingerprint") != self.fp:
                raise SpecError(f"plan fingerprint mismatch: {self.path}; rebuild on this host")
            if loaded.get("id") != fingerprint_id(self.fp):
                raise SpecError(f"plan fingerprint id mismatch: {self.path}")
            if not all(isinstance(loaded.get(k), dict) for k in ("entries", "models")):
                raise SpecError(f"malformed plan entries/models: {self.path}")
            self.data = loaded

    @property
    def entries(self):
        return self.data["entries"]

    def save(self):
        # Atomic publication, not a multi-writer transaction. Callers sharing a
        # plan still need external serialization to avoid lost updates.
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".kurn-plan-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self.data, fh, indent=1, sort_keys=True, allow_nan=False)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def lookup(self, op, fmt, K, N, regime="cold", threads=None):
        threads = (os.cpu_count() or 1) if threads is None else threads
        _positive_int("threads", threads)
        e = self.entries.get(entry_key(op, fmt, K, N, regime, threads))
        if not e:
            return None
        if not isinstance(e, dict) or not isinstance(e.get("config"), dict):
            raise SpecError("malformed plan entry")
        c = resolve({**e["config"], "threads": threads})
        if (c["op"], c["weights"]) != (op, fmt):
            raise SpecError("plan entry config does not match its op/format key")
        return c


def load(path=None):
    return Plan(path)


def lookup(op, fmt, K, N, regime="cold", threads=None, path=None):
    """Resolved config of the stored winner, or None if this shape was never tuned here."""
    return Plan(path).lookup(op, fmt, K, N, regime, threads)


def for_model(model, regime="cold", threads=None, ops=("gemv",), path=None):
    """{tensor name: resolved config} for every tuned tensor of the model (others omitted)."""
    p = Plan(path)
    out = {}
    for name, _tn, fmt, K, N in model_shapes(model):
        for op in ops:
            if fmt:
                c = p.lookup(op, fmt, K, N, regime, threads)
                if c:
                    out[name if len(ops) == 1 else f"{name}:{op}"] = c
    return out


# --------------------------------------------------------------------------- building
def _bench_args(K, N, cols=None):
    a = ["--K", str(K), "--N", str(N)]
    if cols:
        a += ["--M", str(cols)]
    return a


def tune_entry(op, fmt, K, N, regime="cold", threads=None, target=None, objective="energy", n0=27, top=4, secs=0.3,
               reps=3, jobs=3, cols=None, harness=None, log=print, seed=0, refine=True, static_w=0.0):
    """Multi-fidelity tuning of one plan entry. Returns the entry dict."""
    t0 = time.time()
    threads = (os.cpu_count() or 1) if threads is None else threads
    for name, value in (("threads", threads), ("K", K), ("N", N), ("n0", n0), ("top", top), ("reps", reps), ("jobs", jobs)):
        _positive_int(name, value)
    _finite_number("secs", secs, strict=True)
    _finite_number("static_w", static_w)
    if objective not in OBJECTIVES:
        raise SpecError(f"objective {objective!r}: expected one of {list(OBJECTIVES)}")
    if regime not in ("hot", "cold"):
        raise SpecError("regime must be hot or cold")
    target = target or default_target(op, fmt)
    spec = {"op": op, "weights": fmt, "target": target}
    if cols:
        spec["cols"] = cols
    obj = OBJECTIVES[objective]
    resolve({**spec, "threads": threads})  # reject invalid requests before any builds
    slice_n = max(16, ((N + 16 * threads - 1) // (16 * threads)) * 16)
    res, stats = search({**spec, "threads": 1}, None, "hot", objective, n0=n0, keep=top, secs0=0.03, secs=0.15,
                        refine=refine, max_moves=4, extra=_bench_args(K, slice_n, cols), harness=harness, jobs=jobs,
                        seed=seed, log=lambda m: None, static_w=static_w)
    t1 = time.time()
    cands = {}
    try:
        base = resolve({**spec, "threads": threads})
        cands["default"] = base
    except SpecError:
        pass
    for i, r in enumerate(res[:top]):
        c = resolve({**r["config"], "threads": threads})
        key = tuple(c[k] for k in CODEGEN_KEYS)
        if all(tuple(v[k] for k in CODEGEN_KEYS) != key for v in cands.values()):
            cands[f"search{i}"] = c
    sos = {}
    for n, c in cands.items():
        try:
            sos[n] = build(c)
        except BuildError as e:
            log(f"  build failed for {n}: {str(e).splitlines()[0]}")
    meas = {n: [] for n in sos}
    for _ in range(reps):
        for n, so in sos.items():
            try:
                row = bench(so, cands[n], regime, secs, _bench_args(K, N, cols), harness)
                m = measurement(row, static_w)
                reason = None if m is not None else "unusable timing or correctness row"
            except HarnessError as e:
                m, reason = None, str(e)
            if m is None:
                log(f"  bench rejected for {n}: {reason}")
            else:
                meas[n].append(m)
    # A fast median cannot erase a failed correctness check or missing run.
    summ = {n: summarize_measurements(ms) for n, ms in meas.items() if len(ms) == reps}
    if not summ:
        raise HarnessError(f"no candidate passed for {entry_key(op, fmt, K, N, regime, threads)}")
    win = min(summ, key=lambda n: summ[n][obj])
    wc = cands[win]
    entry = {
        "config": {k: wc[k] for k in CODEGEN_KEYS + RUNTIME_KEYS},
        "us": round(summ[win]["us"], 2), "GBps": round(summ[win]["GBps"], 1),
        "energy_uJ": round(summ[win]["energy_uJ"], 1),
        "cpu_us": summ[win]["cpu_us"], "edp": summ[win]["edp"], "relerr": summ[win]["relerr"],
        "static_w": static_w, "reps": reps,
        "default_us": round(summ["default"]["us"], 2) if "default" in summ else None,
        "candidates": {n: round(s["us"], 2) for n, s in summ.items()},
        "winner": win, "objective": objective,
        "search": {"space_active": stats.get("space_active"), "legal_est": stats.get("legal_est"),
                   "builds": stats["builds"], "measurements": stats["measurements"], "secs": round(t1 - t0, 1)},
        "tune_s": round(time.time() - t0, 1),
        "tuned_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return entry


def build_plan(model, threads=None, ops=("gemv",), regime="cold", force=False, path=None, log=print, **kw):
    """Tune every distinct (op, format, shape) of the model that is not in the plan yet."""
    t0 = time.time()
    threads = (os.cpu_count() or 1) if threads is None else threads
    _positive_int("threads", threads)
    p = Plan(path)
    shapes = model_shapes(model)
    todo, skipped = {}, set()
    for name, tn, fmt, K, N in shapes:
        if not fmt:
            skipped.add(tn)
            continue
        for op in ops:
            if (op, fmt) not in KERNELS:
                continue
            k = entry_key(op, fmt, K, N, regime, threads)
            todo.setdefault(k, (op, fmt, K, N, []))[4].append(name)
    if skipped:
        log(f"not kurn formats (left to the host engine): {sorted(skipped)}")
    built = 0
    for k, (op, fmt, K, N, names) in todo.items():
        cached = p.entries.get(k, {})
        compatible = (cached.get("objective") == kw.get("objective", "energy")
                      and cached.get("static_w", 0.0) == kw.get("static_w", 0.0)
                      and cached.get("reps", 0) >= kw.get("reps", 3)
                      and (kw.get("target") is None or cached.get("config", {}).get("target") == kw["target"]))
        if cached and compatible and not force:
            log(f"{k}: cached ({p.entries[k]['us']} us, {p.entries[k]['winner']})")
            continue
        log(f"{k}: tuning ({len(names)} tensors) ...")
        cols = 4 if op == "verify" else None
        e = tune_entry(op, fmt, K, N, regime, threads, cols=cols, log=log, **kw)
        e["tensors"] = len(names)
        p.entries[k] = e
        p.save()
        built += 1
        log(f"  winner {e['winner']}: {e['us']} us ({e['GBps']} GB/s), default {e['default_us']} us, "
            f"tuned in {e['tune_s']} s")
    st = os.stat(model)
    p.data["models"][os.path.basename(model)] = {"size": st.st_size, "entries": sorted(todo), "threads": threads,
                                                  "regime": regime, "ops": list(ops)}
    p.save()
    return p, {"entries": len(todo), "tuned": built, "wall_s": round(time.time() - t0, 1)}


def reuse_cost(model, regime="cold", threads=None, ops=("gemv",), path=None):
    """Seconds to load the plan, map the model's tensors to configs and fetch every kernel
    from the build cache (what a model load pays when the plan exists)."""
    t0 = time.time()
    cfgs = for_model(model, regime, threads, ops, path)
    t1 = time.time()
    sos = {build(c) for c in cfgs.values()}
    return {"tensors": len(cfgs), "kernels": len(sos), "lookup_s": round(t1 - t0, 3), "total_s": round(time.time() - t0, 3)}


# --------------------------------------------------------------------------- CLI
def cmd_plan(a):
    if a.plan_cmd == "build":
        ops = tuple(a.ops.split(","))
        p, st = build_plan(a.model, a.threads, ops, a.regime, a.force, n0=a.n0, top=a.top, jobs=a.jobs,
                           objective=a.objective, harness=a.harness, static_w=a.static_w,
                           reps=a.reps, seed=a.seed)
        rc = reuse_cost(a.model, a.regime, a.threads, ops)
        print(f"plan {p.path}: {st['entries']} entries ({st['tuned']} tuned) in {st['wall_s']} s; "
              f"reuse at load: {rc['tensors']} tensors -> {rc['kernels']} kernels in {rc['total_s']} s")
        return 0
    if a.plan_cmd == "lookup":
        c = lookup(a.op, a.format, a.K, a.N, a.regime, a.threads)
        if c is None:
            print("not in plan")
            return 1
        print(json.dumps({k: c[k] for k in CODEGEN_KEYS}, indent=1))
        return 0
    p = Plan()
    if a.json:
        print(json.dumps(p.data, indent=1, sort_keys=True))
        return 0
    fp = p.data["fingerprint"]
    print(f"plan {p.path}\n  {fp['cpu']}, {fp['cores']} cores, {fp['caches']}, kurn {fp['kurn']}")
    keys = sorted(p.entries)
    if a.model:
        keys = [k for k in keys if k in p.data["models"].get(os.path.basename(a.model), {}).get("entries", ())]
    for k in keys:
        e = p.entries[k]
        cfg = e["config"]
        desc = " ".join(f"{x}={cfg[x]}" for x in ("layout", "rows", "cols", "prefetch", "unpack", "correction", "scales",
                                                   "accum") if x in cfg and cfg[x] not in ("auto",))
        print(f"  {k:40s} {e['us']:9.2f} us {e['GBps']:7.1f} GB/s  (default {e['default_us']} us)  {desc}"
              f"  [tuned {e['tune_s']} s]")
    return 0


def add_plan_cli(sub):
    p = sub.add_parser("plan", help="plan cache: tune once per machine x model shape, reuse at load")
    ps = p.add_subparsers(dest="plan_cmd", required=True)
    b = ps.add_parser("build", help="tune every distinct matmul shape of a GGUF model not yet in the plan")
    b.add_argument("model")
    b.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    b.add_argument("--ops", default="gemv")
    b.add_argument("--regime", default="cold", choices=["hot", "cold"])
    b.add_argument("--objective", default="energy", choices=list(OBJECTIVES))
    b.add_argument("--static-w", type=float, default=0.0, help="platform watts added to the CPU-time energy proxy")
    b.add_argument("--reps", type=int, default=3, help="required successful final repetitions per candidate")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--n0", type=int, default=27, help="initial sample of the staged search per entry")
    b.add_argument("--top", type=int, default=4, help="search survivors re-measured in the target regime")
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--force", action="store_true", help="re-tune entries already in the plan")
    b.add_argument("--harness")
    s = ps.add_parser("show", help="print the plan for this machine")
    s.add_argument("--model")
    s.add_argument("--json", action="store_true")
    q = ps.add_parser("lookup", help="print the stored config for one shape")
    q.add_argument("op")
    q.add_argument("format")
    q.add_argument("K", type=int)
    q.add_argument("N", type=int)
    q.add_argument("--regime", default="cold")
    q.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.set_defaults(fn=cmd_plan)
