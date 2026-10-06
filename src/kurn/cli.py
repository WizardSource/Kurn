"""kurn command line.

kurn check   SPEC [k=v ...]                      validate; print the resolved config and tune space
kurn gen     SPEC [k=v ...] [-o out.c] [--embed PREFIX]
kurn build   SPEC [k=v ...] [--out-dir DIR]      emit + compile a shared library; prints its path
kurn verify  SPEC [k=v ...] [--space | --all]    numerical check against the reference
kurn tune    SPEC [k=v,v ...] [--regime cold] [--objective energy|speed|edp] [--static-w W] [--secs S] [--rounds R]
kurn roofline [--threads T] [--streams S]        measured peak read bandwidth (DRAM and L2)
kurn targets                                     what this host can build and run
"""

import argparse
import json
import os
import shlex
import sys

from . import __version__
from .harness import HarnessError, bandwidth, check
from .kernels import embed, generate
from .spec import CODEGEN_KEYS, TARGETS, SpecError, iter_space, legal_configs, load, parse_overrides, resolve, validate_space
from .toolchain import BuildError, ToolchainError, build, cc_for, run_mode
from .tune import OBJECTIVES, tune


def _spec_args(p):
    p.add_argument("spec", help="path to a .kurn spec")
    p.add_argument("overrides", nargs="*", help="key=value overrides (tune: key=v1,v2 sweeps)")


def _load(a, allow_lists=False):
    spec, space = load(a.spec)
    single, lists = parse_overrides(a.overrides)
    if lists and not allow_lists:
        raise SpecError(f"list overrides ({', '.join(lists)}) are only valid for `kurn tune`")
    return spec, space, single, lists


def cmd_check(a):
    spec, space, ov, _ = _load(a)
    c = resolve(spec, ov)
    print(json.dumps(c, indent=1))
    for k, vs in space.items():
        print(f"tune {k}: {vs}")
    if space:
        n = validate_space(spec, {**space, **{k: [v] for k, v in ov.items()}})
        print(f"tune space: {n} legal configurations")


def cmd_gen(a):
    spec, _, ov, _ = _load(a)
    src = generate(resolve(spec, ov))
    if a.embed:
        src = embed(src, a.embed)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(src)
    else:
        sys.stdout.write(src)


def cmd_build(a):
    spec, _, ov, _ = _load(a)
    print(build(resolve(spec, ov), a.out_dir))


def _verify_one(c, strict):
    label = " ".join(f"{k}={c[k]}" for k in CODEGEN_KEYS)
    so = build(c, extra_flags=("-Wall", "-Wextra", "-Wshadow", "-Werror") if strict else ())
    mode, why = run_mode(c["target"])
    if mode is None:
        print(f"built  {label}  (compiled, not run: {why})")
        return "built", why
    row = check(so, c)
    ok = row["check"] == "ok"
    print(f"{'ok    ' if ok else 'FAIL  '} {label}  relerr={row['relerr']:.1e}{'  [qemu]' if mode == 'qemu' else ''}")
    return ("ok" if ok else "fail"), ""


def cmd_verify(a):
    if a.all:
        configs = list(legal_configs())
    else:
        spec, space, ov, _ = _load(a)
        if a.space:
            seen, configs = set(), []
            for _, c in iter_space(spec, {**space, **{k: [v] for k, v in ov.items()}}):
                key = tuple(c[k] for k in CODEGEN_KEYS)
                if key not in seen:
                    seen.add(key)
                    configs.append(c)
        else:
            configs = [resolve(spec, ov)]
    fails, skipped, notrun = 0, {}, {}
    for c in configs:
        t = c["target"]
        if t in skipped:
            skipped[t] += 1
            continue
        try:
            status, why = _verify_one(c, a.strict)
        except ToolchainError as e:  # no compiler/assembler for this target: report once, skip the rest
            skipped[t] = 1
            print(f"skip   target {t}: {e}")
            continue
        except (BuildError, HarnessError) as e:
            fails += 1
            print(f"FAIL   {c['weights']} {c['op']} {t}: {e}")
            continue
        fails += status == "fail"
        if status == "built":
            notrun.setdefault(t, [0, why])[0] += 1
    tail = ""
    if skipped:
        tail += f", {sum(skipped.values())} skipped (no usable toolchain for {', '.join(skipped)}; reasons above)"
    if notrun:
        tail += f", {sum(n for n, _ in notrun.values())} compiled but not run (" + "; ".join(
            f"{t}: {why}" for t, (_, why) in notrun.items()) + ")"  # fmt: skip
    print(f"{len(configs)} configurations, {fails} failures{tail}")
    return 1 if fails else 0


def cmd_tune(a):
    spec, space, ov, lists = _load(a, allow_lists=True)
    space = {**space, **{k: [v] for k, v in ov.items()}, **lists}
    if not space:
        raise SpecError("nothing to tune: add a `tune` line to the spec or pass key=v1,v2 overrides")
    res, front = tune(spec, space, a.regime, a.objective, a.static_w, a.secs, shlex.split(a.bench_args), a.out, a.harness,
                      rounds=a.rounds, keep=a.keep, budget=a.budget, max_spread=a.max_spread)  # fmt: skip
    if not res:
        print("no configuration passed")
        return 1
    show = ("us", "cpu_us", "energy_uJ", "GBps", "GOPs")
    obj = OBJECTIVES[a.objective]

    def fmt(r):
        keys = " ".join(f"{k}={v}" for k, v in r.items() if k in space)
        vals = "  ".join(f"{k}={r[k]:.1f}" for k in show)
        return f"{keys}  {vals}  (median of {r['rounds']}, spread {r['spread_' + obj]:.1%})"

    print(f"\nbest by {a.objective} (median of interleaved rounds):")
    for r in res[:5]:
        print("  ", fmt(r))
    print("pareto front (time vs energy):")
    for r in front:
        print("  ", fmt(r))
    warnings = res[0].get("warnings") or []
    print(
        "ranking: " + ("NOT resolved -- " + "; ".join(warnings) if warnings else "resolved (leaders' order stable, spread within limits)")
    )
    return 0


def cmd_roofline(a):
    bw = bandwidth(a.threads, a.streams, detail=True)
    for k in ("dram", "l2"):
        if k in bw:
            print(f"{k}: {bw[k]:.1f} GB/s (threads={a.threads}, streams={bw[k + '_streams']}; peak of the best group, "
                  f"median {bw[k + '_median']:.1f} GB/s)")  # fmt: skip


def cmd_targets(a):
    print(f"{'op':5} {'weights':7} {'target':12} {'build':8} run")
    for (op, fmt), targets in TARGETS.items():
        for t in targets:
            try:
                cc_for(t)
                b = "yes"
            except ToolchainError:
                b = "no"  # run_mode gives the reason
            except BuildError:
                b = "no"
            mode, why = run_mode(t)
            print(f"{op:5} {fmt:7} {t:12} {b:8} {mode or 'no: ' + why}")


def cmd_model(a):
    from .model.compile_model import compile_model

    out, arch, c = compile_model(a.gguf, a.out)
    print(f"{out}  ({arch}; Q8_0 kernel: layout={c['layout']} rows={c['rows']})")
    print(f"run: {out} {a.gguf} gen THREADS N_GEN tok,tok,...   |   {out} {a.gguf} ppl THREADS CTX tokens.txt")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kurn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"kurn {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="validate a spec and print the resolved config")
    _spec_args(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("gen", help="emit C for a spec")
    _spec_args(p)
    p.add_argument("-o", "--out")
    p.add_argument("--embed", metavar="PREFIX", help="static, PREFIX-ed symbols, no kurn.h include (paste into ggml)")
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("build", help="emit and compile a shared library")
    _spec_args(p)
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("verify", help="check generated kernels against the reference")
    p.add_argument("spec", nargs="?", help="path to a .kurn spec (omit with --all)")
    p.add_argument("overrides", nargs="*")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--space", action="store_true", help="every legal config in the spec's tune space")
    g.add_argument("--all", action="store_true", help="every legal codegen config for every op/format/target")
    p.add_argument("--strict", action="store_true", help="compile with -Wall -Wextra -Wshadow -Werror")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("tune", help="sweep the tune space; rank by energy proxy, speed or EDP")
    _spec_args(p)
    p.add_argument("-o", "--out", help="write all results as CSV")
    p.add_argument("--regime", default="cold", choices=["hot", "cold"],
                   help="hot: weights cache-resident; cold: stream >1.2 GB from DRAM (default)")  # fmt: skip
    p.add_argument("--objective", default="energy", choices=list(OBJECTIVES))
    p.add_argument("--static-w", type=float, default=0.0, help="platform power charged per wall-second (W)")
    p.add_argument("--secs", type=float, default=1.0, help="length of one measurement (default 1 s)")
    p.add_argument("--rounds", type=int, default=3, help="interleaved measurements per configuration, ranked on the median (default 3)")
    p.add_argument("--keep", type=int, default=3, help="leaders re-measured until their order is stable (default 3)")
    p.add_argument("--budget", type=float, default=30.0, help="seconds of extra measuring for the leaders (default 30)")
    p.add_argument("--max-spread", type=float, default=0.10, help="warn when a leader's interquartile range / median exceeds this (0.10)")
    p.add_argument("--bench-args", default="", help="extra harness args, e.g. '--K 2048 --N 512 --serial-us 20'")
    p.add_argument("--harness", help="use another harness binary with the same CLI (e.g. the ggml-linked one)")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("roofline", help="measure read bandwidth (DRAM and L2)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--streams", type=int, default=0, help="streams per thread (default 0: best of 1, 2, 4, 8)")
    p.set_defaults(fn=cmd_roofline)

    p = sub.add_parser("model", help="compile the whole decode step of one GGUF model (Qwen3 / OLMoE, Q8_0) into one program")
    p.add_argument("gguf")
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_model)

    p = sub.add_parser("targets", help="list targets and whether this host can build/run them")
    p.set_defaults(fn=cmd_targets)

    # --- layout ---
    from .plan import add_plan_cli
    from .tune import add_search_cli

    add_search_cli(sub)
    add_plan_cli(sub)
    # --- end layout ---

    a, rest = ap.parse_known_args(argv)
    for tok in rest:  # allow key=value overrides after options, e.g. `kurn tune s.kurn --regime hot rows=1,2`
        if tok.startswith("-") or "=" not in tok or not hasattr(a, "overrides"):
            ap.error(f"unrecognized argument {tok!r}")
        a.overrides.append(tok)
    if a.cmd == "verify" and not a.all and not a.spec:
        ap.error("verify needs a SPEC (or --all)")
    try:
        return a.fn(a) or 0
    except (SpecError, BuildError, HarnessError) as e:
        print(f"kurn: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

# --- compress ---
# `kurn mix ...` (per-tensor mixed precision, kurn.mixed): dispatched before the core parser.
_main_before_compress = main


def main(argv=None):  # noqa: F811
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["mix"]:
        from ._numpy import have_numpy

        if not have_numpy():
            print("kurn: `kurn mix` needs numpy and gguf: pip install 'kurn[gguf]'", file=sys.stderr)
            return 2
        from .mixed import cli_main

        return cli_main(args[1:])
    return _main_before_compress(argv)


# --- end compress ---


# --- attn ---
_core_main = main


def main(argv=None):  # noqa: F811  (extension commands from kurn.hooks.COMMANDS, e.g. `kurn attn ...`)
    from . import hooks

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in hooks.COMMANDS:
        return hooks.COMMANDS[argv[0]](argv[1:])
    backend = _spec_backend(argv)
    if backend:
        return hooks.TARGET_BACKENDS[backend](argv[0], argv[1:])
    return _core_main(argv)


# --- end attn ---


# --- gpu ---
def _spec_backend(argv):
    """The hooks.TARGET_BACKENDS target named by a spec command's spec file or target=... override."""
    from . import hooks

    if len(argv) < 2 or argv[0] not in ("check", "gen", "build", "verify", "tune") or not hooks.TARGET_BACKENDS:
        return None
    target = next((t.split("=", 1)[1] for t in argv[1:] if t.startswith("target=")), None)
    if target is None:
        path = next((t for t in argv[1:] if not t.startswith("-") and "=" not in t), None)
        if not path or not os.path.isfile(path):
            return None
        try:
            spec, _ = load(path)
        except (OSError, SpecError):
            return None
        target = spec.get("target")
    return target if target in hooks.TARGET_BACKENDS else None


# --- end gpu ---
