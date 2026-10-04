"""Energy-aware autotuner.

Every configuration in the search space is generated, compiled, numerically
checked and timed by the harness, in several interleaved rounds (ranked on the
median; the leaders are re-measured until their order is stable, see `tune`).
Configurations that fail the check are dropped. Results are ranked by an energy
proxy, by speed, or by energy x delay, and the time/energy Pareto front is reported.

Energy proxy (no RAPL needed): busy CPU-time x PROXY_W_PER_CORE, plus an
optional platform/static power term charged per wall-clock second. Under a pure
busy-core proxy, fewer threads can look cheaper even when slower; `static_w`
models the rest of the machine that stays powered while the kernel runs.
"""

import csv
import itertools
import os

from .harness import HarnessError, bench
from .spec import SpecError, resolve
from .toolchain import BuildError, build

PROXY_W_PER_CORE = 5.47
OBJECTIVES = {"energy": "energy_uJ", "speed": "us", "edp": "edp"}


def energy_uj(row, static_w=0.0):
    return row["cpu_us"] * PROXY_W_PER_CORE + row["us"] * static_w


def pareto_front(results):
    front, best_e = [], float("inf")
    for r in sorted(results, key=lambda r: (r["us"], r["energy_uJ"])):
        if r["energy_uJ"] < best_e:
            front.append(r)
            best_e = r["energy_uJ"]
    return front


# Noise-robust ranking (`tune`): one short run per configuration ranks configurations by whatever else the machine was
# doing at that moment. Instead every candidate is measured `rounds` times in interleaved rounds (the order rotates each
# round, so slow drift hits every candidate alike) and ranked on the median. The top `keep` candidates are then re-measured
# in extra interleaved rounds until their order holds for STABLE_ROUNDS consecutive rounds or `budget` seconds of extra
# measuring are spent. A candidate's spread is the interquartile range / median of its samples; when the top two are closer than their
# spread, or a top candidate's spread exceeds `max_spread`, the ranking is reported as not resolved, with a hint.
STABLE_ROUNDS = 2
MAX_EXTRA_ROUNDS = 12


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _summary(samples, static_w):
    """Per-candidate medians of every metric, plus the spread of each objective."""
    med = {k: _median([x[k] for x in samples]) for k in ("us", "cpu_us", "GBps", "GOPs", "relerr")}
    e = [energy_uj(x, static_w) for x in samples]
    edp = [ei * x["us"] for ei, x in zip(e, samples)]
    out = {**med, "energy_uJ": _median(e), "edp": _median(edp), "rounds": len(samples)}
    for key, vals in (("us", [x["us"] for x in samples]), ("energy_uJ", e), ("edp", edp)):
        out[f"spread_{key}"] = _spread(vals)
    return out


def _spread(vals):
    """Interquartile range / median (robust to a single disturbed measurement); 0 for one sample."""
    import statistics

    if len(vals) < 2:
        return 0.0
    q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    m = _median(vals)
    return (q3 - q1) / m if m else 0.0


def rank_warnings(results, objective, max_spread=0.10):
    """Human-readable warnings when the measured spread is too large to trust the ranking of the leaders."""
    key = OBJECTIVES[objective]
    warn = []
    if len(results) >= 2:
        a, b = results[0], results[1]
        gap = (b[key] - a[key]) / a[key] if a[key] else 0.0
        noise = max(a[f"spread_{key}"], b[f"spread_{key}"])
        if gap < noise:
            warn.append(f"top two not resolved: {gap:.1%} apart but their samples spread {noise:.1%}")
    noisy = [r for r in results[:3] if r[f"spread_{key}"] > max_spread]
    if noisy:
        warn.append(f"{len(noisy)} of the top {min(3, len(results))} spread more than {max_spread:.0%} "
                    f"(worst {max(r[f'spread_{key}'] for r in noisy):.1%})")  # fmt: skip
    if warn:
        warn.append("the machine is probably busy: rerun with longer runs (--secs 3) or more rounds (--rounds 5), "
                    "or on a quiet machine")  # fmt: skip
    return warn


def tune(spec, space, regime="cold", objective="energy", static_w=0.0, secs=1.0, extra=(), out=None,
         harness=None, log=print, rounds=3, keep=3, budget=30.0, max_spread=0.10):  # fmt: skip
    """Sweep `space` (dict key -> list of values) on top of `spec`, `rounds` interleaved measurements of `secs` each,
    then refine the leaders (see above). Returns (results ranked by the median objective, pareto front); each result has
    the medians, `rounds` and `spread_*`, and the first carries `warnings` (empty when the ranking is resolved)."""
    import time

    if objective not in OBJECTIVES:
        raise SpecError(f"objective {objective!r}: expected one of {list(OBJECTIVES)}")
    if rounds < 1:
        raise SpecError("rounds must be >= 1")
    key = OBJECTIVES[objective]
    keys = list(space)
    cands = []
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
            cands.append({"ov": ov, "c": c, "so": build(c), "samples": []})
        except BuildError as e:
            log(f"FAIL {ov}: {str(e).splitlines()[0]}")

    def measure(cand, rnd):
        try:
            row = bench(cand["so"], cand["c"], regime, secs, extra, harness)
        except HarnessError as e:
            log(f"FAIL {cand['ov']}: {str(e).splitlines()[0]}")
            cand["dead"] = True
            return
        if row["check"] == "FAIL":
            log(f"FAIL {cand['ov']}: relerr {row['relerr']:.2e} vs reference")
            cand["dead"] = True
            return
        cand["samples"].append({k: row[k] for k in ("us", "cpu_us", "GBps", "GOPs", "relerr")})
        e = energy_uj(row, static_w)
        log(f"r{rnd} {cand['ov']}  {row['us']:9.1f} us  {e:10.1f} uJ  {row['GBps']:7.1f} GB/s  relerr {row['relerr']:.1e}")

    def live():
        return [c for c in cands if not c.get("dead")]

    for rnd in range(rounds):  # interleaved rounds, rotated order
        pool = live()
        for i in range(len(pool)):
            measure(pool[(i + rnd) % len(pool)], rnd)

    def ranked():
        alive = [c for c in live() if c["samples"]]
        for c in alive:
            c["stats"] = _summary(c["samples"], static_w)
        return sorted(alive, key=lambda c: c["stats"][key])

    order = ranked()
    if len(order) > 1 and keep > 1:  # refine the leaders until their order is stable or the budget is spent
        t0, stable, extra_rounds = time.monotonic(), 0, 0
        top = [id(c) for c in order[:keep]]
        while stable < STABLE_ROUNDS and extra_rounds < MAX_EXTRA_ROUNDS and time.monotonic() - t0 < budget:
            lead = order[: min(keep, len(order))]
            for i in range(len(lead)):
                measure(lead[(i + extra_rounds) % len(lead)], rounds + extra_rounds)
            extra_rounds += 1
            order = ranked()
            now = [id(c) for c in order[:keep]]
            stable = stable + 1 if now == top else 0
            top = now
        log(f"refined the top {min(keep, len(order))}: {extra_rounds} extra round(s); order "
            f"{'stable' if stable >= STABLE_ROUNDS else 'still changing (budget or round limit reached)'}")  # fmt: skip
    results = [{**c["ov"], **c["stats"]} for c in order]
    if results:
        results[0]["warnings"] = rank_warnings(results, objective, max_spread)
        for w in results[0]["warnings"]:
            log(f"warning: {w}")
    if out and results:
        cols = [k for k in results[0] if k != "warnings"]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
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

    return generate(c)


def sample_legal(spec, space, n, rng, max_tries=None):
    """Up to n distinct legal configs (deduplicated on the generated C). The space is
    conditional (most keys only matter for some layouts), so sampling is sequential:
    `layout` first, then every other key in random order, each uniformly among the values
    that keep the partial assignment legal (unassigned keys at their defaults).
    Returns (configs, active keys {key: values}, legal fraction of uniform draws over them)."""
    keys = list(space)
    seen, out, tries = set(), [], 0
    max_tries = max_tries or 20 * n
    while len(out) < n and tries < max_tries:
        tries += 1
        ov = {}
        order = [k for k in keys if k == "layout"] + rng.sample([k for k in keys if k != "layout"], len(keys) - ("layout" in keys))
        for k in order:
            ok = []
            for v in space[k]:
                try:
                    resolve(spec, {**ov, k: v})
                    ok.append(v)
                except SpecError:
                    pass
            if ok:
                ov[k] = rng.choice(ok)
        try:
            c = resolve(spec, ov)
        except SpecError:
            continue
        src = _source_key(c)
        if src in seen:
            continue
        seen.add(src)
        out.append((ov, c))
    # effective space: keys that take >= 2 values among legal configs; legal fraction of
    # uniform draws over those keys (the others at their defaults)
    active = {k: space[k] for k in keys if len({repr(c.get(k)) for _, c in out}) > 1}
    legal = 0
    for _ in range(2000):
        try:
            resolve(spec, {k: rng.choice(v) for k, v in active.items()})
            legal += 1
        except SpecError:
            pass
    return out, active, legal / 2000


def _build_all(cs, jobs, log):
    """Build in parallel; returns {index: so path} for the configs that compiled."""
    from concurrent.futures import ThreadPoolExecutor

    def one(ic):
        i, c = ic
        try:
            return i, build(c)
        except BuildError as e:
            log(f"FAIL build {i}: {str(e).splitlines()[0]}")
            return i, None

    with ThreadPoolExecutor(max(1, jobs)) as ex:
        return {i: so for i, so in ex.map(one, enumerate(cs)) if so}


def _measure(c, so, regime, secs, extra, harness, static_w):
    row = bench(so, c, regime, secs, extra, harness)
    if row["check"] == "FAIL":
        return None
    e = energy_uj(row, static_w)
    return {"us": row["us"], "cpu_us": row["cpu_us"], "energy_uJ": e, "edp": e * row["us"], "GBps": row["GBps"], "relerr": row["relerr"]}


def search(
    spec,
    space=None,
    regime="hot",
    objective="energy",
    static_w=0.0,
    n0=81,
    eta=3,
    keep=3,
    secs0=0.05,
    secs=0.4,
    refine=True,
    min_gain=0.02,
    max_moves=12,
    extra=(),
    harness=None,
    jobs=3,
    seed=0,
    log=print,
    start=(),
):
    """Staged search (see above). Returns (ranked results, stats). `start` adds explicit
    override dicts to the initial sample (e.g. a baseline to compare against)."""
    import random
    import time

    if objective not in OBJECTIVES:
        raise SpecError(f"objective {objective!r}: expected one of {list(OBJECTIVES)}")
    obj = OBJECTIVES[objective]
    t0 = time.time()
    if space is None:
        space = search_space(spec["op"], spec["weights"], spec["target"])
    rng = random.Random(seed)
    cand, active, frac = sample_legal(spec, space, n0, rng)
    for ov in start:
        try:
            cand.append((dict(ov), resolve(spec, ov)))
        except SpecError as e:
            log(f"skip start {ov}: {e}")
    stats = {
        "space_keys": {k: len(v) for k, v in space.items()},
        "space_raw": space_size(space),
        "active_keys": {k: len(v) for k, v in active.items()},
        "space_active": space_size(active),
        "legal_fraction": round(frac, 4),
        "legal_est": int(space_size(active) * frac),
        "sampled": len(cand),
        "builds": 0,
        "measurements": 0,
        "rounds": [],
    }
    log(
        f"search space: {len(space)} keys, {stats['space_raw']:,} raw combinations; {len(active)} keys active "
        f"({stats['space_active']:,} combinations, ~{frac:.1%} legal: ~{stats['legal_est']:,}); sampled {len(cand)} distinct"
    )
    sos = _build_all([c for _, c in cand], jobs, log)
    stats["builds"] += len(cand)
    pool = [(ov, c, sos[i]) for i, (ov, c) in enumerate(cand) if i in sos]
    scores = {}
    budget = secs0
    while True:
        rnd = []
        for i, (ov, c, so) in enumerate(pool):
            try:
                m = _measure(c, so, regime, budget, extra, harness, static_w)
            except HarnessError as e:
                log(f"FAIL {ov}: {str(e).splitlines()[0]}")
                m = None
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
        m = _measure(c, so, regime, secs, extra, harness, static_w)
        stats["measurements"] += 1
        if m:
            best.append({"ov": ov, "c": c, "so": so, **m})
    best.sort(key=lambda r: r[obj])
    if refine and best:
        best = _coordinate_descent(spec, space, best, regime, obj, secs, min_gain, max_moves, extra, harness, static_w, jobs, log, stats)
    stats["wall_s"] = round(time.time() - t0, 1)
    res = [
        {
            **r["ov"],
            "us": r["us"],
            "cpu_us": r["cpu_us"],
            "energy_uJ": r["energy_uJ"],
            "edp": r["edp"],
            "GBps": r["GBps"],
            "relerr": r["relerr"],
            "config": {k: r["c"][k] for k in r["c"] if k not in ("act_format",)},
        }
        for r in best
    ]
    res.sort(key=lambda r: r[obj])
    log(f"search done in {stats['wall_s']} s: {stats['builds']} builds, {stats['measurements']} measurements")
    return res, stats


def _confirm(a, b, regime, obj, secs, extra, harness, static_w, reps=3):
    """Interleaved re-measurement of two candidates; returns medians (a, b) of `obj`."""
    import statistics

    ma, mb = [], []
    for _ in range(reps):
        for r, acc in ((a, ma), (b, mb)):
            m = _measure(r["c"], r["so"], regime, secs, extra, harness, static_w)
            acc.append(m[obj] if m else float("inf"))
    return statistics.median(ma), statistics.median(mb)


def _coordinate_descent(spec, space, best, regime, obj, secs, min_gain, max_moves, extra, harness, static_w, jobs, log, stats):
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
                m = _measure(c, sos[i], regime, secs, extra, harness, static_w)
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
                    best.append(inc)
                    inc = ch
                    moves += 1
                    improved = True
            best.extend(t for t in trial if t is not ch)
    return [inc] + sorted((r for r in best if r is not inc), key=lambda r: r[obj])


def cmd_search(a):
    import json
    import shlex

    from .spec import load, parse_overrides

    spec, space0 = load(a.spec)
    single, lists = parse_overrides(a.overrides)
    spec = {**spec, **single}
    space = search_space(
        spec["op"],
        spec["weights"],
        spec["target"],
        keys=a.keys.split(",") if a.keys else None,
        exclude=tuple(single) + tuple(k for k in spec if k not in ("op", "weights", "target")),
    )
    space.update(space0)
    space.update(lists)
    res, stats = search(
        spec,
        space,
        a.regime,
        a.objective,
        a.static_w,
        n0=a.n0,
        eta=a.eta,
        keep=a.keep,
        secs0=a.secs0,
        secs=a.secs,
        refine=not a.no_refine,
        extra=shlex.split(a.bench_args),
        harness=a.harness,
        jobs=a.jobs,
        seed=a.seed,
    )
    if not res:
        print("no configuration passed")
        return 1
    print(f"\nbest by {a.objective}:")
    for r in res[:5]:
        print(
            "  ",
            " ".join(f"{k}={r['config'][k]}" for k in space),
            f"  us={r['us']:.2f} energy_uJ={r['energy_uJ']:.1f} GBps={r['GBps']:.1f}",
        )
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"spec": spec, "space": space, "stats": stats, "results": res}, fh, indent=1, default=str)
    return 0


def add_search_cli(sub):
    p = sub.add_parser(
        "search",
        help="staged search (sampling + successive halving + coordinate descent) over every "
        "legal value of the codegen keys, for spaces too large for `tune`",
    )
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
