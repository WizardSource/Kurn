"""`kurn gpu ...`: the CUDA backend's commands.

kurn gpu targets                                  nvcc, GPU, archs and kernels available here
kurn gpu check  SPEC [k=v ...]                    validate a cuda spec; print the resolved config
kurn gpu gen    SPEC [k=v ...] [-o out.cu]        emit CUDA C++
kurn gpu build  SPEC [k=v ...] [--archs ...]      nvcc -> shared library (kurn_gpu.h ABI); prints path and resources
kurn gpu ptxas  [SPEC | --all] [--archs ...]      registers / shared memory / spills / static occupancy per arch (no GPU)
kurn gpu verify [SPEC | --all] [--space] [--gpu]  numerics vs the exact reference: CPU emulator (default) or the local GPU
kurn gpu sass   [SPEC | --all] [--arch sm_XX]     SASS instruction mix (mma, ldmatrix, cp.async, local memory, hot loop)
kurn gpu harness [--llama DIR] [--arch sm_XX]     build the GPU harness (optionally linked with llama.cpp's ggml-cuda)
kurn gpu info | roofline                          device facts, measured HBM bandwidth and tensor-core peak
kurn gpu tune   SPEC [k=v,v ...] [--brief N]      energy-ranked tuning on the local GPU (NVML joules)
kurn gpu matrix [--formats ...] [--quick]         the benchmark matrix vs every installed competitor -> RESULTS dir
kurn gpu report RESULTS_DIR                       wins/ties/losses report.md + dispatch.json
kurn gpu dispatch FMT BATCH [--table T]           what kernel_for() picks
kurn gpu kit-tune [--quick] [--out tuned.json]    brief tuning of the kernels the matrix races (hand-run kit)
"""

import argparse
import json
import os
import sys

from ..spec import SpecError, parse_overrides
from . import spec as gspec


def _load(a, allow_lists=False):
    sp, space = gspec.load(a.spec)
    single, lists = parse_overrides(a.overrides)
    if lists and not allow_lists:
        raise SpecError(f"list overrides ({', '.join(lists)}) are only valid for `tune`")
    return sp, space, single, lists


def _arch(a):
    from .harness import detect_arch

    return getattr(a, "arch", None) or detect_arch() or "sm_80"


def cmd_targets(a):
    from .harness import detect_arch, gpu_present
    from .toolchain import cxx, nvcc, nvcc_version

    print(f"nvcc: {nvcc() or 'not found'}{' (CUDA ' + nvcc_version() + ')' if nvcc() else ''}")
    from .toolchain import cxx_problem

    print(f"GPU:  {detect_arch() if gpu_present() else 'none (kernels can be built and emulator-verified, not timed)'}")
    print(f"emulator: {'ok (' + ' '.join(cxx()) + ')' if not cxx_problem() else 'unavailable: ' + cxx_problem()}")
    print(f"archs: {', '.join(gspec.ARCHS)}")
    print(f"{'op':5} {'weights':7} {'act':5} {'unit':>4}  doc")
    for f, d in gspec.FORMATS.items():
        for op in ("gemv", "gemm") if d["gemm"] else ("gemv",):
            print(f"{op:5} {f:7} {d['act']:5} {d['unit']:4}  {d['doc']}")


def cmd_check(a):
    sp, space, ov, _ = _load(a)
    c = gspec.resolve(sp, ov)
    print(json.dumps(c, indent=1))
    for k, vs in space.items():
        print(f"tune {k}: {vs}")
    if space:
        print(f"tune space: {gspec.validate_space(sp, {**space, **{k: [v] for k, v in ov.items()}})} legal configurations")


def cmd_gen(a):
    from .codegen import generate

    sp, _, ov, _ = _load(a)
    src = generate(gspec.resolve(sp, ov))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(src)
    else:
        sys.stdout.write(src)


def cmd_build(a):
    from .toolchain import nvcc_build, resource_rows

    sp, _, ov, _ = _load(a)
    c = gspec.resolve(sp, ov)
    archs = a.archs.split(",") if a.archs else [c["arch"]]
    so, rep = nvcc_build(c, archs, out_dir=a.out_dir)
    print(so)
    for r in resource_rows(c, {ar: {k: v for k, v in rep.items() if k.startswith(ar + ":")} for ar in archs}):
        print(f"  {r['arch']}: {r['regs']} registers, {r['smem']} B shared, {r['spill']} B spills, occupancy {r['occupancy']:.0%}")


def _configs(a):
    if getattr(a, "defaults", False):  # every format's default GEMV and default engine tiles (what the matrix races)
        from .matrix import default_kernels

        arch = getattr(a, "arch", None) or "sm_80"
        return [c for f in gspec.FORMATS for c, _, _ in default_kernels(f, arch).values()]
    if a.all:
        out = []
        for f, d in gspec.FORMATS.items():
            for op in ("gemv", "gemm") if d["gemm"] else ("gemv",):
                out += gspec.covering_configs(op, f, extra=a.extra)
        return out
    sp, space, ov, _ = _load(a)
    if a.space:
        seen, out = set(), []
        for _, c in gspec.iter_space(sp, {**space, **{k: [v] for k, v in ov.items()}}):
            if gspec.config_key(c) not in seen:
                seen.add(gspec.config_key(c))
                out.append(c)
        return out
    return [gspec.resolve(sp, ov)]


def cmd_ptxas(a):
    from .toolchain import build_many, ptxas_report, resource_rows

    archs = a.archs.split(",")
    configs = _configs(a)
    bad = 0
    for c, rep in build_many(configs, lambda c: ptxas_report(c, archs)):
        if isinstance(rep, Exception):
            bad += 1
            print(f"FAIL  {c['op']} {c['weights']} {gspec.label(c)}: {str(rep).splitlines()[0]}")
            continue
        rows = resource_rows(c, rep)
        spill = sum(r["spill"] for r in rows)
        bad += spill > 0 and a.strict
        print(f"{'ok   ' if not spill else 'SPILL'} {c['op']} {c['weights']:6} {gspec.label(c)}  " +
              "  ".join(f"{r['arch']}: r{r['regs']} s{r['smem']} occ{r['occupancy']:.0%}" + (f" spill{r['spill']}" if r["spill"] else "")
                        for r in rows))  # fmt: skip
    print(f"{len(configs)} configurations, {bad} failures")
    return 1 if bad else 0


def cmd_verify(a):
    configs = _configs(a)
    if a.gpu:
        from .harness import build_harness, run_kernel
        from .toolchain import build_many, nvcc_build

        arch = _arch(a)
        h = build_harness(arch)
        fails = 0
        for c, lib in build_many([dict(c, arch=arch) for c in configs], lambda c: nvcc_build(c)[0]):
            if isinstance(lib, Exception):
                fails += 1
                print(f"FAIL   {c['op']} {c['weights']} {gspec.label(c)}: {str(lib).splitlines()[0]}")
                continue
            n, k, m = (4096 + 37, 4096, 1) if c["op"] == "gemv" else (4096 + 48, 4096, 77)  # the engine needs N % 16 == 0
            if c["op"] == "gemv" and c["cols"] > 1:
                m = 2 * c["cols"] + 1
            try:
                chk, _ = run_kernel(h, lib, c["weights"], n, k, m, reps=0, secs=0.01, cold=0)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL   {c['op']} {c['weights']} {gspec.label(c)}: {e}")
                continue
            ok = chk["status"] == "ok"
            fails += not ok
            tag = "ok    " if ok else "FAIL  "
            print(f"{tag} {c['op']} {c['weights']:6} {gspec.label(c)}  relerr={chk['relerr_exact']:.1e}  [gpu {arch}]")
        print(f"{len(configs)} configurations, {fails} failures")
        return 1 if fails else 0
    from .emu import verify
    from .toolchain import cxx_problem

    if cxx_problem():
        print(f"skip   CPU emulator: {cxx_problem()}")
        print(f"{len(configs)} configurations, 0 failures, {len(configs)} skipped (CPU emulator unavailable)")
        return 0
    fails = verify(configs, jobs=a.jobs)
    print(f"{len(configs)} configurations, {fails} failures (CPU emulator)")
    return 1 if fails else 0


def cmd_sass(a):
    from .sass import inspect, summary

    bad = 0
    for c in _configs(a):
        try:
            rep = inspect(c, a.arch)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {c['op']} {c['weights']} {gspec.label(c)}: {str(e).splitlines()[0]}")
            continue
        main = rep.get("kg_gemm" if c["op"] == "gemm" else "kg_gemv", {})
        local = main.get("counts", {}).get("local", 0)
        bad += bool(local)
        print(f"{'LOCAL' if local else 'ok   '} {c['op']} {c['weights']:6} {gspec.label(c)}\n      {summary(c, rep)}")
    return 1 if bad else 0


def cmd_harness(a):
    from .harness import build_harness

    print(build_harness(_arch(a), a.llama, a.out_dir))


def cmd_info(a):
    from .harness import build_harness, info, roofline

    h = a.harness or build_harness(_arch(a))
    print(json.dumps(roofline(h) if a.cmd == "roofline" else info(h), indent=1))


def cmd_tune(a):
    from .harness import build_harness
    from .tune import pareto_front, tune

    sp, space, ov, lists = _load(a, allow_lists=True)
    space = {**space, **{k: [v] for k, v in ov.items()}, **lists}
    if not space:
        from .tune import BRIEF

        space = dict(BRIEF[sp.get("op", "gemv")])
    arch = _arch(a)
    h = a.harness or build_harness(arch)
    shape = tuple(int(x) for x in a.shape.split("x"))
    res, front = tune(sp, space, h, shape, a.objective, a.secs, a.reps, a.out, sample=a.brief, arch=arch)
    if not res:
        print("no configuration passed")
        return 1
    print(f"\nbest by {a.objective}:")
    for r in res[:5]:
        print(f"   {gspec.label(r['config'])}  us={r['us']:.2f}±{r['us_sd']:.2f}  uJ={r['J'] * 1e6:.2f}  GB/s={r['gbps']:.0f}")
    print("pareto front (time vs energy):")
    for r in pareto_front(res):
        print(f"   {gspec.label(r['config'])}  us={r['us']:.2f}  uJ={r['J'] * 1e6:.2f}")
    return 0


def cmd_matrix(a):
    from . import matrix
    from .harness import build_harness, info, roofline

    arch = _arch(a)
    os.makedirs(a.results, exist_ok=True)
    h = a.harness or build_harness(arch, a.llama)
    fmts = a.formats.split(",") if a.formats else list(gspec.FORMATS)
    with open(os.path.join(a.results, "info.json"), "w") as fh:
        json.dump(info(h), fh)
    with open(os.path.join(a.results, "roofline.json"), "w") as fh:
        json.dump(roofline(h), fh)
    kernels = {f: matrix.default_kernels(f, arch) for f in fmts}
    if a.kernels:
        with open(a.kernels) as fh:
            tuned = json.load(fh)  # {fmt: {name: [config string, lo, hi]}}
        from .dispatch import parse_config

        for f, d in tuned.items():
            for n, (cfg, lo, hi) in d.items():
                kernels.setdefault(f, {})[n] = (dict(parse_config(cfg), arch=arch), lo, hi)
    libs = matrix.build_kernels({f: kernels[f] for f in fmts if f in kernels})
    batches = (1, 16) if a.quick else matrix.BATCHES
    plan = matrix.write_plan(os.path.join(a.results, "plan.txt"), libs, fmts, batches=batches,
                             reps=3 if a.quick else a.reps, secs=0.2 if a.quick else a.secs)  # fmt: skip
    rc = matrix.run(h, plan, os.path.join(a.results, "matrix.jsonl"), os.path.join(a.results, "matrix.log"))
    from .report import write

    md, _ = write(a.results, fmts)
    print(md)
    return rc


def cmd_report(a):
    from .report import write

    md, _ = write(a.results, a.formats.split(",") if a.formats else None, dry=a.dry)
    print(md)


def cmd_dispatch(a):
    from .dispatch import kernel_for

    r = kernel_for(a.fmt, a.batch, a.arch, a.table)
    print(json.dumps(r, indent=1, default=str))


def _spec_args(p, lists=False):
    p.add_argument("spec")
    p.add_argument("overrides", nargs="*")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["kit-tune"]:
        from .kit import main as kit_main

        return kit_main(argv[1:])
    ap = argparse.ArgumentParser(prog="kurn gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("targets")
    p.set_defaults(fn=cmd_targets)
    p = sub.add_parser("check")
    _spec_args(p)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("gen")
    _spec_args(p)
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_gen)
    p = sub.add_parser("build")
    _spec_args(p)
    p.add_argument("--archs")
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_build)
    for name, fn in (("ptxas", cmd_ptxas), ("verify", cmd_verify), ("sass", cmd_sass)):
        p = sub.add_parser(name)
        p.add_argument("spec", nargs="?")
        p.add_argument("overrides", nargs="*")
        p.add_argument("--all", action="store_true", help="covering set of every format and op")
        p.add_argument("--defaults", action="store_true", help="the default kernels the benchmark matrix races")
        p.add_argument("--space", action="store_true", help="every legal config in the spec's tune space")
        p.add_argument("--extra", type=int, default=24, help="random configs per (op, format) on top of the covering set")
        p.add_argument("--jobs", type=int)
        p.set_defaults(fn=fn)
        if name == "ptxas":
            p.add_argument("--archs", default="sm_80,sm_90,sm_100")
            p.add_argument("--strict", action="store_true", help="count register spills as failures")
        elif name == "sass":
            p.add_argument("--arch", default="sm_80")
        else:
            p.add_argument("--gpu", action="store_true", help="run on the local GPU instead of the CPU emulator")
            p.add_argument("--arch")
    p = sub.add_parser("harness")
    p.add_argument("--llama", help="llama.cpp checkout built with -DGGML_CUDA=ON (adds the ggml-cuda competitor)")
    p.add_argument("--arch")
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_harness)
    for name in ("info", "roofline"):
        p = sub.add_parser(name)
        p.add_argument("--harness")
        p.add_argument("--arch")
        p.set_defaults(fn=cmd_info)
    p = sub.add_parser("tune")
    _spec_args(p)
    p.add_argument("--objective", default="energy", choices=["energy", "speed", "edp"])
    p.add_argument("--shape", default="4096x14336x1", help="NxKxM to tune on")
    p.add_argument("--secs", type=float, default=0.3)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--brief", type=int, help="random sample of N configs, then re-time the leaders")
    p.add_argument("--harness")
    p.add_argument("--arch")
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_tune)
    p = sub.add_parser("matrix")
    p.add_argument("--results", default="kurn-gpu-results")
    p.add_argument("--formats")
    p.add_argument("--kernels", help="JSON {fmt: {name: [config, mmin, mmax]}} of tuned KURN kernels")
    p.add_argument("--llama")
    p.add_argument("--harness")
    p.add_argument("--arch")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--secs", type=float, default=0.5)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_matrix)
    p = sub.add_parser("report")
    p.add_argument("results")
    p.add_argument("--formats")
    p.add_argument("--dry", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("dispatch")
    p.add_argument("fmt")
    p.add_argument("batch", type=int)
    p.add_argument("--arch")
    p.add_argument("--table")
    p.set_defaults(fn=cmd_dispatch)
    a = ap.parse_args(argv)
    if a.cmd == "verify" and not a.all and not a.spec and not a.defaults:
        ap.error("verify needs a SPEC (or --all)")
    if a.cmd in ("ptxas", "sass") and not a.all and not a.spec and not a.defaults:
        ap.error(f"{a.cmd} needs a SPEC (or --all)")
    from .harness import HarnessError
    from .toolchain import GpuBuildError

    try:
        return a.fn(a) or 0
    except (SpecError, GpuBuildError, HarnessError) as e:
        print(f"kurn gpu: {e}", file=sys.stderr)
        return 2


# `kurn check|gen|build|verify|tune SPEC` on a spec with `target cuda` (see kurn.hooks.TARGET_BACKENDS)
SPEC_COMMANDS = ("check", "gen", "build", "verify", "tune")


def spec_command(cmd, argv):
    return main([cmd, *argv])
