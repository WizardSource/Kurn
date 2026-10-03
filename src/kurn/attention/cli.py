"""kurn attn: command line for the attention op.

kurn attn check   SPEC [k=v ...]            validate; print the resolved config
kurn attn gen     SPEC [k=v ...] [-o out.c] emit C
kurn attn build   SPEC [k=v ...]            emit + compile; prints the shared library path
kurn attn verify  SPEC [k=v ...] [--space]  check on awkward shapes against the float64 reference
kurn attn bench   SPEC [k=v ...] [--regime cold] [--secs S]   time the spec's problem
kurn attn tune    SPEC [k=v,v ...] [--objective energy|speed|edp] [-o out.csv]
kurn attn peak    [--threads T]             AMX-BF16 / AVX-512 BF16 / FP32 FMA peak GFLOP/s
"""

import argparse
import json
import subprocess
import sys

from .. import attention as A
from ..spec import SpecError, parse_overrides
from ..toolchain import BuildError


def _load(a, allow_lists=False):
    spec, space = A.load(a.spec)
    single, lists = parse_overrides(a.overrides)
    if lists and not allow_lists:
        raise SpecError(f"list overrides ({', '.join(lists)}) are only valid for `kurn attn tune`")
    return spec, space, single, lists


def cmd_check(a):
    spec, space, ov, _ = _load(a)
    print(json.dumps(A.resolve(spec, ov), indent=1))
    for k, vs in space.items():
        print(f"tune {k}: {vs}")


def cmd_gen(a):
    spec, _, ov, _ = _load(a)
    src = A.generate(A.resolve(spec, ov))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(src)
    else:
        sys.stdout.write(src)


def cmd_build(a):
    spec, _, ov, _ = _load(a)
    print(A.build(A.resolve(spec, ov)))


def cmd_verify(a):
    spec, space, ov, _ = _load(a)
    configs = [A.resolve(spec, ov)]
    if a.space:
        import itertools

        space = {**space, **{k: [v] for k, v in ov.items()}}
        configs, seen = [], set()
        for combo in itertools.product(*space.values()):
            try:
                c = A.resolve(spec, dict(zip(space, combo)))
            except SpecError:
                continue
            key = tuple(c[k] for k in A.CODEGEN_KEYS)
            if key not in seen:
                seen.add(key)
                configs.append(c)
    fails = 0
    for c in configs:
        label = " ".join(f"{k}={c[k]}" for k in A.CODEGEN_KEYS)
        row = A.check(A.build(c, extra_flags=("-Wall", "-Wextra", "-Werror") if a.strict else ()), c)
        ok = row["check"] == "ok"
        fails += not ok
        print(f"{'ok    ' if ok else 'FAIL  '} {label}  relerr={row['relerr']:.1e} (tol {A.TOL[c['target']]:.0e})")
    print(f"{len(configs)} configurations, {fails} failures")
    return 1 if fails else 0


def cmd_bench(a):
    spec, _, ov, _ = _load(a)
    c = A.resolve(spec, ov)
    row = A.bench(A.build(c), c, a.regime, a.secs, ("--check-toks", "4"))
    for k in (
        "n_q",
        "n_kv",
        "heads",
        "kv_heads",
        "threads",
        "regime",
        "us",
        "cpu_us",
        "GFLOPs",
        "GBps",
        "proxy_uJ_per_call",
        "relerr",
        "check",
    ):
        print(f"{k:18} {row[k]}")
    return 0 if row["check"] == "ok" else 1


def cmd_tune(a):
    spec, space, ov, lists = _load(a, allow_lists=True)
    space = {**space, **{k: [v] for k, v in ov.items()}, **lists}
    if not space:
        raise SpecError("nothing to tune: add a `tune` line to the spec or pass key=v1,v2 overrides")
    res, front = A.tune(spec, space, a.regime, a.objective, a.static_w, a.secs, a.out)
    if not res:
        print("no configuration passed")
        return 1
    show = lambda r: " ".join(f"{k}={r[k]}" for k in space) + f"  us={r['us']:.1f} energy_uJ={r['energy_uJ']:.1f} GFLOPs={r['GFLOPs']:.1f}"
    print(f"\nbest by {a.objective}:")
    for r in res[:5]:
        print("  ", show(r))
    print("pareto front (time vs energy):")
    for r in front:
        print("  ", show(r))
    return 0


def cmd_peak(a):
    return subprocess.run([A.build_harness(), "--peak", "--threads", str(a.threads)]).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kurn attn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def spec_cmd(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("spec")
        p.add_argument("overrides", nargs="*")
        p.set_defaults(fn=fn)
        return p

    spec_cmd("check", cmd_check, "validate a spec")
    spec_cmd("gen", cmd_gen, "emit C").add_argument("-o", "--out")
    spec_cmd("build", cmd_build, "emit and compile")
    p = spec_cmd("verify", cmd_verify, "check against the float64 reference")
    p.add_argument("--space", action="store_true")
    p.add_argument("--strict", action="store_true", help="compile with -Wall -Wextra -Werror")
    p = spec_cmd("bench", cmd_bench, "time the spec's problem")
    p.add_argument("--regime", default="hot", choices=["hot", "cold"])
    p.add_argument("--secs", type=float, default=1.0)
    p = spec_cmd("tune", cmd_tune, "sweep the tune space")
    p.add_argument("-o", "--out")
    p.add_argument("--regime", default="hot", choices=["hot", "cold"])
    p.add_argument("--objective", default="energy", choices=["energy", "speed", "edp"])
    p.add_argument("--static-w", type=float, default=0.0)
    p.add_argument("--secs", type=float, default=1.0)
    p = sub.add_parser("peak", help="peak AMX / AVX-512 BF16 / FMA throughput")
    p.add_argument("--threads", type=int, default=1)
    p.set_defaults(fn=cmd_peak)
    a, rest = ap.parse_known_args(argv)
    for tok in rest:
        if tok.startswith("-") or "=" not in tok or not hasattr(a, "overrides"):
            ap.error(f"unrecognized argument {tok!r}")
        a.overrides.append(tok)
    try:
        return a.fn(a) or 0
    except (SpecError, BuildError, A.HarnessError) as e:
        print(f"kurn attn: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
