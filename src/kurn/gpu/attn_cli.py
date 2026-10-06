"""`kurn gpu attn ...`: the CUDA attention op. Tiers: sm_80 (A100, measured), sm_100 (B200/GB200) and sm_120 (RTX 50),
the last two compile-only so far.

kurn gpu attn check   SPEC [k=v ...]                 validate; print the resolved config, shared memory, archs it fits
kurn gpu attn gen     SPEC [k=v ...] [-o out.cu]     emit CUDA C++
kurn gpu attn build   SPEC [k=v ...]                 nvcc fatbin (sm_80 + sm_100 + sm_120 SASS, PTX); path and resources
kurn gpu attn ptxas   [SPEC | --all [--tier T]]      registers / spills per arch, fatbin contents (no GPU)
kurn gpu attn verify  [SPEC | --all [--tier T]] [--quick] [--gpu]   numerics vs float64: CPU emulator, or the local GPU
kurn gpu attn bench   SPEC [k=v ...]                 check + time the spec's problem on the local GPU (cold regime)
kurn gpu attn tune    SPEC [k=v,v ...]               correctness-gated sweep on the local GPU (speed or NVML energy)
kurn gpu attn matrix  [--results DIR] [--quick]      decode matrix: model shapes x context x KV format, default kernels
kurn gpu attn report  RESULTS_DIR                    markdown table of RESULTS_DIR/attn_matrix.jsonl
kurn gpu attn harness [--arch sm_80]                 build the GPU harness; prints its path

A spec is a `.kurn` file with `op attn` and `target cuda` (see kurn.gpu.attn); SPEC may be `-` for the defaults.
"""

import argparse
import json
import os
import sys

from ..spec import SpecError, parse_overrides
from . import attn as A
from .toolchain import GpuBuildError


def _load(a, allow_lists=False):
    spec, space = ({}, {}) if a.spec in (None, "-") else A.load(a.spec)
    single, lists = parse_overrides(a.overrides)
    if lists and not allow_lists:
        raise SpecError(f"list overrides ({', '.join(lists)}) are only valid for `tune`")
    return spec, space, single, lists


def _local_arch(a):
    from .harness import detect_arch

    return getattr(a, "arch", None) or detect_arch() or A.ARCH


def _configs(a, default_tiers=A.ARCHS):
    tier = getattr(a, "tier", None)
    tiers = A.ARCHS if tier == "all" else (tier,) if tier else default_tiers
    if getattr(a, "all", False):
        return [c for t in tiers for c in A.covering_configs(arch=t)]
    if getattr(a, "defaults", False):  # what kernel_for() dispatches per tier
        return [A.kernel_for(kv, dk, t) for t in tiers for kv in A.KV_FORMATS for dk in A.HEAD_DIMS]
    spec, space, ov, _ = _load(a)
    return [A.resolve(spec, ov)]


def cmd_check(a):
    spec, space, ov, _ = _load(a)
    c = A.resolve(spec, ov)
    print(json.dumps(c, indent=1))
    print(f"shared memory: {A.smem_bytes(c)} B per CTA; fits: {', '.join(A.archs_for(c))}")
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
    c = A.resolve(spec, ov)
    so, _ = A.nvcc_build(c)
    print(so)
    for arch, ks in A.ptxas(c).items():
        print(f"  {arch}: kga_main {ks['kga_main']['regs']} registers, {ks['kga_main']['spill']} B spills; shared {A.smem_bytes(c)} B")


def cmd_ptxas(a):
    from .toolchain import build_many

    bad = 0
    res = build_many(_configs(a), lambda c: (A.ptxas(c), A.fatbin_contents(A.nvcc_build(c)[0])))
    for c, r in res:
        if isinstance(r, Exception):
            bad += 1
            print(f"FAIL  {A.label(c)}: {str(r).splitlines()[0]}")
            continue
        rep, (sass, ptx) = r
        spill = sum(k["spill"] for ar in rep.values() for k in ar.values())
        bad += bool(spill)
        cells = "  ".join(f"{ar}: r{ks['kga_main']['regs']}" + (f" spill{ks['kga_main']['spill']}" if ks["kga_main"]["spill"] else "")
                          for ar, ks in rep.items())  # fmt: skip
        print(f"{'SPILL' if spill else 'ok   '} {A.label(c)}  {cells}  smem {A.smem_bytes(c)} fits {'+'.join(A.archs_for(c))}"
              f"  sass {'+'.join(sass)} ptx {'+'.join(ptx)}")  # fmt: skip
    print(f"{len(res)} configurations, {bad} failures")
    return 1 if bad else 0


def cmd_verify(a):
    fails = 0
    if a.gpu:
        arch = _local_arch(a)
        configs = _configs(a, (A.tier_for(arch),))
        print(f"GPU arch {arch} -> tier {A.tier_for(arch)}: {len(configs)} configurations")
        h = A.build_harness(arch)
        for c in configs:
            try:
                w = A.gpu_check(h, c, A.QUICK_SHAPES if a.quick else A.CHECK_SHAPES)
                ok = w["status"] == "ok"
                tag = "ok    " if ok else "FAIL  "
                print(f"{tag} {A.label(c)}  relerr={w['relerr']:.1e} (tol {A.TOL[c['kv']]:.0e})  [gpu {w['device']}]")
            except (GpuBuildError, A.HarnessError) as e:
                ok = False
                print(f"FAIL   {A.label(c)}: {str(e).splitlines()[0]}")
            fails += not ok
    else:
        from .toolchain import cxx_problem

        configs = _configs(a)
        if cxx_problem():
            print(f"skip   CPU emulator: {cxx_problem()}")
            return 0
        for c in configs:
            try:
                w = A.emu_check(c, A.QUICK_SHAPES if a.quick else A.CHECK_SHAPES, scheds=(0,) if a.quick else (0, 7))
                ok = w["ok"]
                print(f"{'ok    ' if ok else 'FAIL  '} {A.label(c)}  relerr={w['relerr']:.1e} (tol {A.TOL[c['kv']]:.0e})  [emu]")
            except (GpuBuildError, A.EmuError) as e:
                ok = False
                print(f"FAIL   {A.label(c)}: {str(e).splitlines()[0]}")
            fails += not ok
    print(f"{len(configs)} configurations, {fails} failures ({'GPU' if a.gpu else 'CPU emulator'})")
    return 1 if fails else 0


def cmd_bench(a):
    spec, _, ov, _ = _load(a)
    c = A.resolve(spec, ov)
    h = a.harness or A.build_harness(_local_arch(a))
    chk, samples = A.gpu_run(h, A.nvcc_build(c)[0], c, secs=a.secs, reps=a.reps, cold_bytes=a.cold_bytes)
    print(json.dumps(chk))
    if samples:
        print(json.dumps(A.summarize(samples)))
    return 0 if chk["status"] == "ok" else 1


def cmd_tune(a):
    spec, space, ov, lists = _load(a, allow_lists=True)
    space = {**space, **{k: [v] for k, v in ov.items()}, **lists} or {"tk": [32, 64, 128], "wn": [1, 2, 4], "split": [0, 8, 16, 32]}
    h = a.harness or A.build_harness(_local_arch(a))
    res = A.tune(spec, space, h, a.objective, a.secs, a.reps, a.cold_bytes)
    if not res:
        print("no configuration passed")
        return 1
    print(f"\nbest by {a.objective}:")
    for r in res[:5]:
        print(f"   {A.label(r['config'])}  us={r['us']:.2f}±{r['us_sd']:.2f}  GB/s={r['GBps']:.0f}  uJ={r['uJ']:.1f}")
    return 0


def cmd_matrix(a):
    arch = _local_arch(a)
    h = a.harness or A.build_harness(arch)
    ctx = (1024, 16384) if a.quick else A.MATRIX_CONTEXTS
    rows = A.matrix(h, a.results, contexts=ctx, secs=0.2 if a.quick else a.secs, reps=3 if a.quick else a.reps, arch=arch)
    bad = sum(r["status"] != "ok" for r in rows)
    print(f"{len(rows)} cells, {bad} not ok -> {a.results}/attn_matrix.jsonl")
    return 1 if bad else 0


def cmd_report(a):
    path = os.path.join(a.results, "attn_matrix.jsonl")
    with open(path) as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    bad = sum(r["status"] != "ok" for r in rows)
    print("# kurn GPU attention: decode matrix\n")
    ran = sorted({(r.get("device", "?"), r.get("sm"), r.get("native")) for r in rows if r["status"] != "error"}, key=str)
    for dev, sm, native in ran:
        how = "native SASS" if native else "JIT from embedded PTX" if native is not None else "?"
        print(f"Ran on: {dev} (sm_{sm}, tier {A.tier_for(f'sm_{sm}') if sm else '?'}, {how})\n")
    print(
        f"{len(rows)} cells, {len(rows) - bad} correct, {bad} not ok. Default (untuned) kernels; times are the cold-KV mean per call "
        "and are informational: this is a correctness run.\n"
    )
    print("| model | kv | context | nq | status | relerr | splits | us | GB/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] == "error":
            print(f"| {r['model']} | {r['kv']} | {r['nkv']} | {r['nq']} | error: {r['error'][:60]} | | | | |")
            continue
        print(
            f"| {r['model']} | {r['kv']} | {r['nkv']} | {r['nq']} | {r['status']} | {r['relerr']:.1e} | {r['splits']} | "
            f"{r['us']:.1f} ± {r['us_sd']:.1f} | {r['GBps']:.0f} |"
        )
    return 1 if bad else 0


def cmd_harness(a):
    print(A.build_harness(_local_arch(a)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kurn gpu attn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def spec_cmd(name, fn, optional=False):
        p = sub.add_parser(name)
        p.add_argument("spec", nargs="?" if optional else None, default="-")
        p.add_argument("overrides", nargs="*")
        p.set_defaults(fn=fn)
        return p

    spec_cmd("check", cmd_check)
    spec_cmd("gen", cmd_gen).add_argument("-o", "--out")
    spec_cmd("build", cmd_build)
    p = spec_cmd("ptxas", cmd_ptxas, optional=True)
    p.add_argument("--all", action="store_true", help="covering set of every KV format and head dim")
    p.add_argument("--tier", choices=[*A.ARCHS, "all"], help="covering set of one tier (default: all three)")
    p.add_argument("--defaults", action="store_true", help="the per-tier default kernels kernel_for() dispatches")
    p = spec_cmd("verify", cmd_verify, optional=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--tier", choices=[*A.ARCHS, "all"], help="covering set of one tier (default: emulator all, GPU its own)")
    p.add_argument("--defaults", action="store_true", help="the per-tier default kernels kernel_for() dispatches")
    p.add_argument("--quick", action="store_true", help="5 shapes, one schedule")
    p.add_argument("--gpu", action="store_true", help="run on the local GPU instead of the CPU emulator")
    p.add_argument("--arch")
    for name, fn in (("bench", cmd_bench), ("tune", cmd_tune)):
        p = spec_cmd(name, fn)
        p.add_argument("--harness")
        p.add_argument("--arch")
        p.add_argument("--secs", type=float, default=0.4)
        p.add_argument("--reps", type=int, default=3)
        p.add_argument("--cold-bytes", type=float, default=3e8, help="KV bytes rotated per timing pass (0: hot, one layer)")
        if name == "tune":
            p.add_argument("--objective", default="speed", choices=["speed", "energy"])
    p = sub.add_parser("matrix")
    p.add_argument("--results", default="kurn-gpu-attn-results")
    p.add_argument("--harness")
    p.add_argument("--arch")
    p.add_argument("--secs", type=float, default=0.4)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_matrix)
    p = sub.add_parser("report")
    p.add_argument("results")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("harness")
    p.add_argument("--arch")
    p.set_defaults(fn=cmd_harness)
    a, rest = ap.parse_known_args(argv)
    for tok in rest:
        if tok.startswith("-") or "=" not in tok or not hasattr(a, "overrides"):
            ap.error(f"unrecognized argument {tok!r}")
        a.overrides.append(tok)
    try:
        return a.fn(a) or 0
    except (SpecError, GpuBuildError, A.HarnessError, A.EmuError) as e:
        print(f"kurn gpu attn: {e}", file=sys.stderr)
        return 2
