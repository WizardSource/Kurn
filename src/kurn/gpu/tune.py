"""Energy-ranked tuning on a real GPU.

Same idea as kurn.tune, with real energy: every configuration is generated, compiled by nvcc,
checked against the exact reference on the GPU (failures are dropped) and timed; the energy per
call is the NVML board-energy counter delta over the timed window (joules, not a proxy). Ranking
by energy, speed or EDP; the time/energy Pareto front is reported.

`brief` mode (the hand-run kit): a random sample of the brief space plus the defaults, timed once,
then the leaders re-timed with more rounds.
"""

import csv
import random
import statistics

from ..spec import SpecError
from .harness import HarnessError, run_kernel
from .spec import config_key, iter_space, label, resolve
from .toolchain import GpuBuildError, build_many, nvcc_build

OBJECTIVES = ("energy", "speed", "edp")

# Brief spaces per op: the keys that matter most for a first look on a new GPU.
BRIEF = {
    "gemv": {"layout": ["native", "split"], "tpr": [8, 16, 32, 64], "rpb": [1, 2, 4, 8], "sub": [1, 2, 4], "unroll": [1, 2, 4],
             "minb": [0, 4]},
    "gemm": {"layout": ["native", "split"], "bm": [64, 128], "bn": [32, 64, 128], "wm": [2, 4], "wn": [1, 2, 4], "bkb": [1, 2],
             "pipe": ["reg2", "async2", "async3"], "pad": [16]},
}  # fmt: skip


def _summ(samples):
    us = [s["us"] for s in samples]
    j = [s["joules"] for s in samples if s.get("joules") == s.get("joules")]  # drop NaN
    return {
        "us": statistics.mean(us),
        "us_sd": statistics.stdev(us) if len(us) > 1 else 0.0,
        "J": statistics.mean(j) if j else float("nan"),
        "gbps": statistics.mean(s["bytes"] / (s["us"] * 1e-6) / 1e9 for s in samples),
    }


def _score(r, objective):
    e = r["J"] if r["J"] == r["J"] else r["us"]  # without NVML energy, fall back to time
    return {"energy": e, "speed": r["us"], "edp": e * r["us"]}[objective]


def pareto_front(results):
    front, best = [], float("inf")
    for r in sorted(results, key=lambda r: (r["us"], r["J"])):
        if r["J"] < best:
            front.append(r)
            best = r["J"]
    return front


def tune(spec, space, harness, shape=(4096, 14336, 1), objective="energy", secs=0.3, reps=3, out=None, log=print,
         jobs=None, sample=None, keep=4, seed=0, arch=None):  # fmt: skip
    """Sweep `space` on top of `spec` on the local GPU. Returns (ranked results, pareto front)."""
    if objective not in OBJECTIVES:
        raise SpecError(f"objective {objective!r}: expected one of {list(OBJECTIVES)}")
    cands, seen = [], set()
    for ov, c in iter_space(spec, space):
        if arch:
            c = dict(c, arch=arch)
        if config_key(c) not in seen:
            seen.add(config_key(c))
            cands.append((ov, c))
    d = resolve(spec)
    if arch:
        d = dict(d, arch=arch)
    if sample and len(cands) > sample:
        rng = random.Random(seed)
        cands = rng.sample(cands, sample)
        if config_key(d) not in {config_key(c) for _, c in cands}:
            cands.append(({}, d))
    n, k, m = shape
    log(f"tune {spec.get('op')}/{spec.get('weights')} on N={n} K={k} M={m}: {len(cands)} configurations, objective {objective}")
    built = dict(zip(range(len(cands)), (r for _, r in build_many([c for _, c in cands], nvcc_build, jobs))))
    results = []
    for i, (ov, c) in enumerate(cands):
        lib = built[i]
        if isinstance(lib, Exception):
            log(f"FAIL build {label(c)}: {str(lib).splitlines()[0]}")
            continue
        lib, rep = lib
        spill = sum(r["spill_st"] + r["spill_ld"] for r in rep.values())
        if spill:  # register spills to local memory: skip before spending GPU time on it
            log(f"skip {label(c)}: {spill} bytes of register spills (ptxas)")
            continue
        try:
            check, samples = run_kernel(harness, lib, c["weights"], n, k, m, reps=1 if sample else reps, secs=secs)
        except (HarnessError, GpuBuildError) as e:
            log(f"FAIL {label(c)}: {e}")
            continue
        if check["status"] != "ok":
            log(f"FAIL {label(c)}: {check['status']} (relerr {check['relerr_exact']:.2e})")
            continue
        r = {"config": c, "lib": lib, "relerr": check["relerr_exact"], **_summ(samples), **ov}
        results.append(r)
        log(f"{label(c)}  {r['us']:9.2f} us  {r['J'] * 1e6:9.2f} uJ  {r['gbps']:7.1f} GB/s  relerr {r['relerr']:.1e}")
    if sample and results:  # re-time the leaders with more rounds, interleaved
        results.sort(key=lambda r: _score(r, objective))
        lead = results[:keep]
        acc = {id(r): [] for r in lead}
        for _ in range(reps):
            for r in lead:
                _, s = run_kernel(harness, r["lib"], r["config"]["weights"], n, k, m, reps=1, secs=secs)
                acc[id(r)] += s
        for r in lead:
            r.update(_summ(acc[id(r)]))
        results = lead + results[keep:]
    results.sort(key=lambda r: _score(r, objective))
    if out and results:
        keys = sorted({k2 for r in results for k2 in r["config"]} - {"kernel", "target"})
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys + ["us", "us_sd", "J", "gbps", "relerr", "lib"])
            for r in results:
                w.writerow([r["config"].get(k2) for k2 in keys] + [r["us"], r["us_sd"], r["J"], r["gbps"], r["relerr"], r["lib"]])
    return results, pareto_front(results)
