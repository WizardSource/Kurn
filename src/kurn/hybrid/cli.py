"""`kurn hybrid ...`: choose CPU, GPU, or both for a matmul / decode step.

kurn hybrid modes                              list devices and modes
kurn hybrid route FMT BATCH [--mode M]         where one matmul runs
kurn hybrid plan  PLAN.json [--mode M]         assign a step (CPU + GPU together)
"""

import argparse
import json
import sys

from .route import MODES, plan, route


def cmd_modes(a):
    from kurn.gpu.harness import detect_arch, gpu_present

    print(f"modes: {', '.join(MODES)}")
    print(f"GPU:   {detect_arch() if gpu_present() else 'none'}")
    print("env:   KURN_DEVICE, KURN_GPU_DISPATCH, KURN_HYBRID_MIN_K")
    print("cpu    — always CPU kernels")
    print("gpu    — always CUDA when a GPU is present")
    print("auto   — one device per op (GPU only where dispatch.json shows a win)")
    print("hybrid — same rule, but a step may use CPU and GPU together")


def cmd_route(a):
    r = route(a.fmt, a.batch, mode=a.mode, arch=a.arch, table=a.table, K=a.K)
    print(json.dumps(r, indent=1, default=str))


def cmd_plan(a):
    with open(a.plan) as fh:
        ops = json.load(fh)
    if isinstance(ops, dict) and "ops" in ops:
        ops = ops["ops"]
    out = plan(ops, mode=a.mode, arch=a.arch, table=a.table)
    print(json.dumps(out, indent=1, default=str))
    if a.mode == "hybrid" or out["mode"] == "hybrid":
        print(f"# {out['summary']}", file=sys.stderr)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(prog="kurn hybrid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("modes")
    p.set_defaults(fn=cmd_modes)
    p = sub.add_parser("route")
    p.add_argument("fmt")
    p.add_argument("batch", type=int)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--arch")
    p.add_argument("--table")
    p.add_argument("-K", type=int)
    p.set_defaults(fn=cmd_route)
    p = sub.add_parser("plan")
    p.add_argument("plan", help="JSON list of {fmt, batch, K?, N?, name?} or {ops: [...]}")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--arch")
    p.add_argument("--table")
    p.set_defaults(fn=cmd_plan)
    a = ap.parse_args(argv)
    try:
        return a.fn(a) or 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"kurn hybrid: {e}", file=sys.stderr)
        return 2
