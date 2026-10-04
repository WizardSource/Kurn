"""Energy-aware autotuner.

Every configuration in the search space is generated, compiled, numerically
checked and timed by the harness. Configurations that fail the check are
dropped. Results are ranked by an energy proxy, by speed, or by energy x delay,
and the time/energy Pareto front is reported.

Energy proxy (no RAPL needed): busy CPU-time x PROXY_W_PER_CORE, plus an
optional platform/static power term charged per wall-clock second. Under a pure
busy-core proxy, fewer threads can look cheaper even when slower; `static_w`
models the rest of the machine that stays powered while the kernel runs.
"""

import csv
import itertools
import math
import os
import statistics

from .harness import HarnessError, bench
from .spec import RUNTIME_KEYS, SpecError, resolve
from .toolchain import BuildError, build

_BASE_BUILD = build

PROXY_W_PER_CORE = 5.47
OBJECTIVES = {"energy": "energy_uJ", "speed": "us", "edp": "edp"}


def energy_uj(row, static_w=0.0):
    return row["cpu_us"] * PROXY_W_PER_CORE + row["us"] * static_w


METRIC_KEYS = ("us", "cpu_us", "energy_uJ", "edp", "GBps")


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpecError(f"{name} must be an integer >= {minimum}")


def _finite_number(name, value, minimum=0.0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (strict and value == minimum)):
        op = ">" if strict else ">="
        raise SpecError(f"{name} must be finite and {op} {minimum}")


def measurement(row, static_w=0.0):
    """Normalize one usable timing row, or return None; energy remains a proxy.

    The existing harness admits 'approx' only below relative error 1e-2. Never
    accept unknown statuses, nonfinite values, negative times, or overflowed EDP.
    A rounded zero CPU time is permitted; zero latency cannot rank a benchmark.
    """
    if row.get("check") not in ("ok", "approx"):
        return None
    try:
        for key in ("us", "cpu_us", "GBps", "relerr"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return None
        if row["us"] <= 0 or row["relerr"] >= 1e-2:
            return None
        e = energy_uj(row, static_w)
        edp = e * row["us"]
        if not math.isfinite(e) or e < 0 or not math.isfinite(edp) or edp < 0:
            return None
        return {"us": row["us"], "cpu_us": row["cpu_us"], "energy_uJ": e, "edp": edp,
                "GBps": row["GBps"], "relerr": row["relerr"]}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def summarize_measurements(samples):
    """Medians of paired observations, not products of marginal medians.

    Correctness uses the worst error, not a median that can hide a bad run.
    Callers must reject the entire candidate if any requested repetition failed.
    """
    if not samples:
        raise HarnessError("no valid measurement samples")
    out = {k: statistics.median(m[k] for m in samples) for k in METRIC_KEYS if all(k in m for m in samples)}
    if all("relerr" in m for m in samples):
        out["relerr"] = max(m["relerr"] for m in samples)
    return out


def pareto_front(results):
    front, best_e = [], float("inf")
    for r in sorted(results, key=lambda r: (r["us"], r["energy_uJ"])):
        if r["energy_uJ"] < best_e:
            front.append(r)
            best_e = r["energy_uJ"]
    return front


def tune(spec, space, regime="cold", objective="energy", static_w=0.0, secs=1.0, extra=(), out=None,
         harness=None, log=print):  # fmt: skip
    """Sweep `space` (dict key -> list of values) on top of `spec`. Returns (ranked results, pareto front)."""
    if objective not in OBJECTIVES:
        raise SpecError(f"objective {objective!r}: expected one of {list(OBJECTIVES)}")
    _finite_number("static_w", static_w)
    _finite_number("secs", secs, strict=True)
    keys = list(space)
    results = []
    for combo in itertools.product(*(space[k] for k in keys)):
        ov = dict(zip(keys, combo))
        try:
            c = resolve(spec, ov)
        except SpecError as e:
            log(f"skip {ov}: {e}")
            continue
        if c["threads"] > (os.cpu_count() or 1):
            log(f"skip {ov}: threads={c['threads']} exceeds the {os.cpu_count()} CPUs here")
            continue
        try:
            row = bench(build(c), c, regime, secs, extra, harness)
        except (BuildError, HarnessError) as e:
            log(f"FAIL {ov}: {str(e).splitlines()[0]}")
            continue
        m = measurement(row, static_w)
        if m is None:
            log(f"FAIL {ov}: unusable timing or correctness row")
            continue
        e = m["energy_uJ"]
        results.append({**ov, **m, "GOPs": row["GOPs"]})
        log(f"{ov}  {row['us']:9.1f} us  {e:10.1f} uJ  {row['GBps']:7.1f} GB/s  relerr {row['relerr']:.1e}")
    results.sort(key=lambda r: r[OBJECTIVES[objective]])
    if out and results:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    return results, pareto_front(results)


# --------------------------------------------------------------------------- staged search
# For spaces too large to sweep (layout=composed: lanes x plane x kblock x rgroup x ... x the
# algorithm keys is ~1e9 raw combinations for one kernel):
#   1. the space is derived from the registry: every legal value of every codegen key for
#      (op, weights, target), `auto` excluded (no hand-picked starting point or hints);
#   2. random legal configurations (rejection sampling, deduplicated on the generated C);
#   3. successive halving: time all at a short budget, keep the best 1/eta, re-time the
#      survivors with eta x the budget (interleaved rounds), until `keep` remain;
#   4. coordinate descent from the survivors: one key at a time, every other legal value;
#      a move is accepted only if it wins an interleaved re-measurement by > `min_gain`.
# Builds run in parallel (`jobs`); timing is always serial.

SEARCH_SKIP = ("threads", "wait", "op", "weights", "target")


def search_space(op, weights, target, keys=None, exclude=()):
    """Every legal value of every codegen key (no `auto`): the space a search starts from."""
    from . import hooks
    from .spec import CODEGEN_KEYS, SCHEDULE

    out = {}
    for k in CODEGEN_KEYS:
        if k in SEARCH_SKIP or k in exclude or (keys and k not in keys):
            continue
        vals = [v for v in SCHEDULE[k](op, weights, target) if v != "auto" and v != hooks.AUTO_VALUES.get(k, "auto")]
        if len(vals) > 1:
            out[k] = vals
    return out


def space_size(space):
    n = 1
    for v in space.values():
        n *= len(v)
    return n


def _source_key(c):
    from .kernels import generate

    # The same C is not the same experiment when target, thread count or wait
    # policy differs. Runtime distinctions must survive candidate deduplication.
    return (c["target"], *(c[k] for k in RUNTIME_KEYS), generate(c))


def sample_legal(spec, space, n, rng, max_tries=None):
    """Up to n distinct legal configs (deduplicated on the generated C). The space is
    conditional (most keys only matter for some layouts), so sampling is sequential:
    `layout` first, then every other key in random order, each uniformly among the values
    that keep the partial assignment legal (unassigned keys at their defaults).
    Returns (configs, active keys {key: values}, legal fraction of uniform draws over them)."""
    _positive_int("n", n)
    if max_tries is not None:
        _positive_int("max_tries", max_tries)
    if any(not values for values in space.values()):
        raise SpecError("search space values must not be empty")
    keys = list(space)
    seen, resolved_seen, out, tries = set(), set(), [], 0
    max_tries = 20 * n if max_tries is None else max_tries
    # Partial legality is deterministic for a fixed spec and registry. Memoize
    # its Boolean result locally, with a fixed memory bound. Final configs and
    # legality-fraction probes retain their original draws. Never cache globally.
    partial_valid = {}

    def is_valid(ov):
        key = tuple(sorted(ov.items()))
        if key in partial_valid:
            return partial_valid[key]
        try:
            resolve(spec, ov)
            valid = True
        except SpecError:
            valid = False
        if len(partial_valid) < 4096:
            partial_valid[key] = valid
        return valid

    while len(out) < n and tries < max_tries:
        tries += 1
        ov = {}
        order = [k for k in keys if k == "layout"] + rng.sample([k for k in keys if k != "layout"], len(keys) - ("layout" in keys))
        for k in order:
            ok = []
            for v in space[k]:
                if is_valid({**ov, k: v}):
                    ok.append(v)
            if ok:
                ov[k] = rng.choice(ok)
        try:
            c = resolve(spec, ov)
        except SpecError:
            continue
        # Local only: no persistent memoization across extension/registry changes.
        # Keep drawing in the same order, preserving the RNG state and semantics.
        resolved_key = tuple(sorted(c.items()))
        if resolved_key in resolved_seen:
            continue
        resolved_seen.add(resolved_key)
        src = _source_key(c)
        if src in seen:
            continue
        seen.add(src)
        out.append((ov, c))
    # effective space: keys that take >= 2 values among legal configs; legal fraction of
    # uniform draws over those keys (the others at their defaults)
    active = {k: space[k] for k in keys if len({repr(c.get(k)) for _, c in out}) > 1}
    # With no active keys, every one of the 2,000 probes is the same resolve and
    # consumes no randomness. Evaluate that exact null case once.
    if not active:
        try:
            resolve(spec)
            return out, active, 1.0
        except SpecError:
            return out, active, 0.0
    # Reuse is worthwhile in small spaces. Keep the original loop for large
    # spaces, where the 2,000 draws may be almost entirely distinct.
    if space_size(active) < 2000:
        return out, active, _probe_legal_fraction(spec, active, rng)
    legal = 0
    for _ in range(2000):
        try:
            resolve(spec, {k: rng.choice(v) for k, v in active.items()})
            legal += 1
        except SpecError:
            pass
    return out, active, legal / 2000


def _probe_legal_fraction(spec, active, rng):
    """Keep every draw, but validate each distinct assignment only once.

    Like the sampler's partial-legality cache, this requires stable spec/registry
    validation within one call. The cache is local and bounded by 2,000 probes.
    Cache False as well as True; repeated invalid draws still count in the
    denominator. Drawing before each lookup preserves the random stream.
    """
    keys, values = tuple(active), tuple(active.values())
    valid = {}
    legal = 0
    for _ in range(2000):
        draw = tuple(rng.choice(v) for v in values)
        if draw not in valid:
            try:
                resolve(spec, dict(zip(keys, draw)))
            except SpecError:
                valid[draw] = False
            else:
                valid[draw] = True
        legal += valid[draw]
    return legal / 2000


def _build_all(cs, jobs, log):
    """Build in parallel, sharing identical built-in compile requests per batch.

    Source sharing is not candidate sharing: thread/wait variants keep their
    indices and are still measured separately. Compiler settings and generators
    must be stable during this call. Custom builders retain the original path.
    """
    from concurrent.futures import ThreadPoolExecutor

    if build is not _BASE_BUILD:
        def custom_one(ic):
            i, c = ic
            try:
                return i, build(c)
            except BuildError as e:
                log(f"FAIL build {i}: {str(e).splitlines()[0]}")
                return i, None

        with ThreadPoolExecutor(max(1, jobs)) as ex:
            return {i: so for i, so in ex.map(custom_one, enumerate(cs)) if so}

    from . import toolchain

    groups = {}
    for i, c in enumerate(cs):
        try:
            src = toolchain.generate(c)
        except BuildError as e:
            log(f"FAIL build {i}: {str(e).splitlines()[0]}")
            continue
        stem = f"{c['weights']}_{c['op']}_{c['target']}"
        key = (c["target"], stem, src)
        groups.setdefault(key, []).append(i)

    def one(item):
        (target, stem, src), indices = item
        try:
            return indices, toolchain.compile_source(src, target, stem=stem)
        except BuildError as e:
            for i in indices:
                log(f"FAIL build {i}: {str(e).splitlines()[0]}")
            return indices, None

    paths = {}
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        for indices, so in ex.map(one, groups.items()):
            if so:
                paths.update((i, so) for i in indices)
    # Group insertion order need not be candidate order (e.g. 0, 2, then 1).
    return {i: paths[i] for i in sorted(paths)}


def _measure(c, so, regime, secs, extra, harness, static_w):
    return measurement(bench(so, c, regime, secs, extra, harness), static_w)


def _safe_measure(c, so, regime, secs, extra, harness, static_w, log=lambda _: None):
    try:
        m = _measure(c, so, regime, secs, extra, harness, static_w)
    except HarnessError as e:
        log(f"FAIL measurement: {str(e).splitlines()[0]}")
        return None
    if m is None:
        log("FAIL measurement: unusable timing or correctness row")
    return m


def search(spec, space=None, regime="hot", objective="energy", static_w=0.0, n0=81, eta=3, keep=3, secs0=0.05,
           secs=0.4, refine=True, min_gain=0.02, max_moves=12, extra=(), harness=None, jobs=3, seed=0, log=print,
           start=()):
    """Staged search (see above). Returns (ranked results, stats). `start` adds explicit
    override dicts to the initial sample (e.g. a baseline to compare against)."""
    import random
    import time

    if objective not in OBJECTIVES:
        raise SpecError(f"objective {objective!r}: expected one of {list(OBJECTIVES)}")
    for name, value, minimum in (("n0", n0, 1), ("eta", eta, 2), ("keep", keep, 1),
                                 ("jobs", jobs, 1), ("max_moves", max_moves, 0)):
        _positive_int(name, value, minimum)
    _finite_number("secs0", secs0, strict=True)
    _finite_number("secs", secs, strict=True)
    _finite_number("static_w", static_w)
    _finite_number("min_gain", min_gain)
    if min_gain >= 1:
        raise SpecError("min_gain must be < 1")
    obj = OBJECTIVES[objective]
    t0 = time.time()
    if space is None:
        space = search_space(spec["op"], spec["weights"], spec["target"])
    rng = random.Random(seed)
    cand, active, frac = sample_legal(spec, space, n0, rng)
    candidate_keys = {_source_key(c) for _, c in cand} if start else set()
    for ov in start:
        try:
            c = resolve(spec, ov)
            key = _source_key(c)
            if key not in candidate_keys:
                candidate_keys.add(key)
                cand.append((dict(ov), c))
        except SpecError as e:
            log(f"skip start {ov}: {e}")
    stats = {"space_keys": {k: len(v) for k, v in space.items()}, "space_raw": space_size(space),
             "active_keys": {k: len(v) for k, v in active.items()}, "space_active": space_size(active),
             "legal_fraction": round(frac, 4), "legal_est": int(space_size(active) * frac), "sampled": len(cand),
             "builds": 0, "measurements": 0, "rounds": []}
    log(f"search space: {len(space)} keys, {stats['space_raw']:,} raw combinations; {len(active)} keys active "
        f"({stats['space_active']:,} combinations, ~{frac:.1%} legal: ~{stats['legal_est']:,}); sampled {len(cand)} distinct")
    sos = _build_all([c for _, c in cand], jobs, log)
    stats["builds"] += len(cand)
    pool = [(ov, c, sos[i]) for i, (ov, c) in enumerate(cand) if i in sos]
    scores = {}
    budget = secs0
    while True:
        rnd = []
        for i, (ov, c, so) in enumerate(pool):
            m = _safe_measure(c, so, regime, budget, extra, harness, static_w, log)
            stats["measurements"] += 1
            if m is None:
                continue
            scores.setdefault(i, []).append(m)
            rnd.append((m[obj], i))
        rnd.sort()
        stats["rounds"].append({"configs": len(pool), "secs": budget, "best": rnd[0][0] if rnd else None})
        log(f"round: {len(pool)} configs at {budget:.2f}s -> best {rnd[0][0]:.2f} ({obj})" if rnd else "round: none passed")
        if len(rnd) <= keep or not rnd:
            pool = [pool[i] for _, i in rnd]
            break
        nkeep = max(keep, len(rnd) // eta)
        pool = [pool[i] for _, i in rnd[:nkeep]]
        scores = {}
        budget = min(secs, budget * eta)
    best = []
    for ov, c, so in pool:
        m = _safe_measure(c, so, regime, secs, extra, harness, static_w, log)
        stats["measurements"] += 1
        if m:
            best.append({"ov": ov, "c": c, "so": so, **m})
    best.sort(key=lambda r: r[obj])
    refined = bool(refine and best)
    if refined:
        best = _coordinate_descent(spec, space, best, regime, obj, secs, min_gain, max_moves, extra, harness, static_w,
                                   jobs, log, stats)
    stats["wall_s"] = round(time.time() - t0, 1)
    res = [{**r["ov"], "us": r["us"], "cpu_us": r["cpu_us"], "energy_uJ": r["energy_uJ"], "edp": r["edp"],
            "GBps": r["GBps"], "relerr": r["relerr"], "config": {k: r["c"][k] for k in r["c"] if k not in ("act_format",)}}
           for r in best]
    # Refinement has already selected the confirmed incumbent.
    # Sorting stale alternatives here can undo that decision.
    if not refined:
        res.sort(key=lambda r: r[obj])
    log(f"search done in {stats['wall_s']} s: {stats['builds']} builds, {stats['measurements']} measurements")
    return res, stats


def _confirm(a, b, regime, obj, secs, extra, harness, static_w, reps=3):
    """Interleaved confirmation; one failed repetition disqualifies a candidate.

    Update all metric fields from the same sample set so the returned objective
    cannot disagree with stale energy/latency fields. Keep the helper's
    (objective median, objective median) return shape.
    """
    _positive_int("reps", reps)
    samples = ([], [])
    for _ in range(reps):
        for r, acc in zip((a, b), samples):
            acc.append(_safe_measure(r["c"], r["so"], regime, secs, extra, harness, static_w))
    medians = []
    for r, ms in zip((a, b), samples):
        r["_valid"] = all(m is not None for m in ms)
        if r["_valid"]:
            summary = summarize_measurements(ms)
            r.update(summary)
            medians.append(summary[obj])
        else:
            medians.append(float("inf"))
    return tuple(medians)


def _coordinate_descent(spec, space, best, regime, obj, secs, min_gain, max_moves, extra, harness, static_w, jobs, log,
                        stats):
    inc = best[0]
    seen = {_source_key(r["c"]) for r in best}
    moves = 0
    improved = True
    while improved and moves < max_moves:
        improved = False
        for k, vals in space.items():
            neigh = []
            for v in vals:
                if v == inc["ov"].get(k, inc["c"].get(k)):
                    continue
                ov = {**inc["ov"], k: v}
                try:
                    c = resolve(spec, ov)
                except SpecError:
                    continue
                src = _source_key(c)
                if src in seen:
                    continue
                seen.add(src)
                neigh.append((ov, c))
            if not neigh:
                continue
            sos = _build_all([c for _, c in neigh], jobs, log)
            stats["builds"] += len(neigh)
            trial = []
            for i, (ov, c) in enumerate(neigh):
                if i not in sos:
                    continue
                m = _safe_measure(c, sos[i], regime, secs, extra, harness, static_w, log)
                stats["measurements"] += 1
                if m:
                    trial.append({"ov": ov, "c": c, "so": sos[i], **m})
            if not trial:
                continue
            ch = min(trial, key=lambda r: r[obj])
            if ch[obj] < inc[obj] * (1 - min_gain):
                a, b = _confirm(inc, ch, regime, obj, secs, extra, harness, static_w)
                stats["measurements"] += 6
                if b < a * (1 - min_gain):
                    log(f"move {k}={ch['ov'][k]}: {a:.2f} -> {b:.2f} ({obj})")
                    ch[obj] = b
                    if not any(r is inc for r in best):
                        best.append(inc)
                    inc = ch
                    moves += 1
                    improved = True
            # Screened neighbors are not eligible without confirmation.
            if not inc.get("_valid", True):
                raise HarnessError("incumbent failed a required confirmation repetition")
            if moves >= max_moves:
                break
    return [inc] + sorted((r for r in best if r is not inc and r.get("_valid", True)), key=lambda r: r[obj])


def cmd_search(a):
    import json
    import shlex

    from .spec import load, parse_overrides

    spec, space0 = load(a.spec)
    single, lists = parse_overrides(a.overrides)
    spec = {**spec, **single}
    space = search_space(spec["op"], spec["weights"], spec["target"], keys=a.keys.split(",") if a.keys else None,
                         exclude=tuple(single) + tuple(k for k in spec if k not in ("op", "weights", "target")))
    space.update(space0)
    space.update(lists)
    res, stats = search(spec, space, a.regime, a.objective, a.static_w, n0=a.n0, eta=a.eta, keep=a.keep, secs0=a.secs0,
                        secs=a.secs, refine=not a.no_refine, extra=shlex.split(a.bench_args), harness=a.harness,
                        jobs=a.jobs, seed=a.seed)
    if not res:
        print("no configuration passed")
        return 1
    print(f"\nbest by {a.objective}:")
    for r in res[:5]:
        print("  ", " ".join(f"{k}={r['config'][k]}" for k in space), f"  us={r['us']:.2f} energy_uJ={r['energy_uJ']:.1f}"
              f" GBps={r['GBps']:.1f}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"spec": spec, "space": space, "stats": stats, "results": res}, fh, indent=1, default=str)
    return 0


def add_search_cli(sub):
    p = sub.add_parser("search", help="staged search (sampling + successive halving + coordinate descent) over every "
                                      "legal value of the codegen keys, for spaces too large for `tune`")
    p.add_argument("spec", help="path to a .kurn spec (op, weights, target; keys set here stay fixed)")
    p.add_argument("overrides", nargs="*", help="k=v fixes a key, k=v1,v2 restricts its values")
    p.add_argument("--keys", help="comma-separated keys to search (default: every codegen key)")
    p.add_argument("--regime", default="hot", choices=["hot", "cold"])
    p.add_argument("--objective", default="energy", choices=list(OBJECTIVES))
    p.add_argument("--static-w", type=float, default=0.0)
    p.add_argument("--n0", type=int, default=81, help="initial random sample")
    p.add_argument("--eta", type=int, default=3)
    p.add_argument("--keep", type=int, default=3)
    p.add_argument("--secs0", type=float, default=0.05)
    p.add_argument("--secs", type=float, default=0.4)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--jobs", type=int, default=3, help="parallel builds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bench-args", default="")
    p.add_argument("--harness")
    p.add_argument("-o", "--out", help="write results + search statistics as JSON")
    p.set_defaults(fn=cmd_search)
