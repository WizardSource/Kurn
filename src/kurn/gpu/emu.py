"""Numerical verification of generated CUDA kernels on the CPU emulator (no GPU needed).

For one config and shape: random ggml weights and f32 activations are written to a scratch
directory, the kernel runs under data/kurn_cuemu.h (repack, quantize, GEMV/GEMM, plus the ggml
activation blocks from kg_xblocks), then
  - the activation blocks must equal ggml's reference quantizer byte for byte, and
  - Y must match the exact double-precision reference on those blocks (relative error <= TOL).
"""

import os
import shutil
import struct
import subprocess
import tempfile

from . import ref
from .spec import FORMATS
from .toolchain import GpuBuildError, emu_build

TOL = 1e-5


class EmuError(Exception):
    pass


def default_shape(c):
    """A small shape that exercises row tails, column tails and several K steps."""
    f = FORMATS[c["weights"]]
    k = max(512, 2 * f["block"]) if f["block"] < 256 else 512
    if c["op"] == "gemm":
        return 72, 256 if c["bkb"] < 4 else 320, 21  # N not a multiple of bm, M not of bn, K tail for bkb=4
    # every lane of a row gets work, and unrolled iterations run past the end of K
    need = c["tpr"] * f["unit"] * c["unroll"] // c["sub"] * 3 // 2
    k = max(k, min(16384, need))
    blk = max(f["block"], 256 if f["act"] == "q8_K" else 32)
    k = (k + blk - 1) // blk * blk
    rows = max(2 * c["rpb"] + 3, 11) if k <= 4096 else c["rpb"] + 3
    return rows, k, max(1, c["cols"] + (1 if c["cols"] > 1 else 0))


def run(c, n=None, k=None, m=None, seed=0, extreme=False, keep=False, sched=0, python_ref=False):
    """Run one config on the emulator. `sched` > 0 randomizes the thread schedule (KEMU_SEED).
    Returns a dict with relerr, quant_ok and the shape."""
    dn, dk, dm = default_shape(c)
    n, k, m = n or dn, k or dk, m or dm
    W, x = ref.problem(c["weights"], n, k, m, seed, extreme)
    exe = emu_build(c)
    d = tempfile.mkdtemp(prefix="kurn-emu-")
    try:
        with open(os.path.join(d, "params.txt"), "w") as fh:
            fh.write(f"{n} {k} {m} {c['weights']}\n")
        with open(os.path.join(d, "W.bin"), "wb") as fh:
            fh.write(W)
        with open(os.path.join(d, "X.bin"), "wb") as fh:
            fh.write(ref.pack_f32(x))
        env = dict(os.environ, KEMU_SEED=str(sched)) if sched else None
        r = subprocess.run([exe, d], capture_output=True, text=True, timeout=600, env=env)
        if r.returncode:
            raise EmuError(f"emulator run failed (exit {r.returncode}): {r.stderr.strip()[-2000:]}")
        with open(os.path.join(d, "Y.bin"), "rb") as fh:
            y = list(struct.unpack(f"<{m * n}f", fh.read()))
        with open(os.path.join(d, "Xb.bin"), "rb") as fh:
            xb = fh.read()
        with open(os.path.join(d, "R.bin"), "rb") as fh:
            rf = list(struct.unpack(f"<{m * n}d", fh.read()))
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)
    want = ref.act_blocks(c["weights"], x)
    quant_ok = xb == want
    if python_ref:
        rf = ref.reference(c["weights"], W, xb, n, k, m)
    err = ref.relerr(y, rf)
    return {"relerr": err, "quant_ok": quant_ok, "ok": quant_ok and err <= TOL, "n": n, "k": k, "m": m}


def verify(configs, log=print, jobs=None, extreme=True):
    """Build in parallel, run each config (random data, and extreme data if `extreme`). Returns failures."""
    from .spec import label
    from .toolchain import build_many

    built = build_many(configs, emu_build, jobs)
    fails = 0
    for c, exe in built:
        tag = f"{c['op']} {c['weights']:6} {label(c)}"
        if isinstance(exe, Exception):
            fails += 1
            log(f"FAIL   {tag}: {str(exe).splitlines()[0]}")
            continue
        try:
            res = [run(c, seed=1)]
            if extreme:
                res += [run(c, seed=2, extreme=True, sched=7), run(c, seed=3, sched=13)]
        except (EmuError, GpuBuildError, subprocess.TimeoutExpired) as e:
            fails += 1
            log(f"FAIL   {tag}: {e}")
            continue
        ok = all(r["ok"] for r in res)
        fails += not ok
        worst = max(r["relerr"] for r in res)
        q = "" if all(r["quant_ok"] for r in res) else "  QUANT MISMATCH"
        log(f"{'ok    ' if ok else 'FAIL  '} {tag}  relerr={worst:.1e}{q}  [emu]")
    return fails
