"""Predicted activation skipping for SwiGLU FFNs (Deja Vu / PowerInfer style), Q8_0, decode.

`data/sparse_ffn.c` computes y = W_down (silu(W_gate x) * (W_up x)) over the active rows only:
mode "gate" computes the whole gate product and skips up/down rows where |silu(g)| < thr;
mode "pred" picks rows with a rank-r predictor (|silu(B A x)| < thr) and skips gate rows too;
mode "dense" is the same code with every row active. `ks_ffn` uses ggml-style (ith, nth)
threading; W_down is stored transposed (row i = column i of W_down, Q8_0 blocks along d).

    from kurn import sparse
    sparse.bench("gate", density=0.5)          # -> harness CSV row as a dict
    sparse.build_lib()                          # -> shared library exposing ks_ffn / ks_workspace
"""

import hashlib
import os
import subprocess
import tempfile

from . import toolchain
from .attention import _compile

MODES = ("dense", "gate", "pred")
COLUMNS = ("impl,mode,regime,threads,d,dff,rank,density,calls,wall_s,cpu_s,us_per_call,GBps,proxy_uJ_per_call,relerr,check,"
           "drift_s").split(",")  # fmt: skip
FLAGS = ("-O3", "-march=x86-64-v4", "-mavx512vnni", "-mf16c")


def _src(name):
    with open(toolchain.data_path(name)) as fh:
        return fh.read()


def _sig():
    return hashlib.sha1(_src("sparse_ffn.c").encode()).hexdigest()[:12]


def build_lib(prefetch=2):
    return _compile(f"/* sparse_ffn.c {_sig()} */\n#define KS_PF {prefetch}\n" + _src("sparse_ffn.c"), FLAGS, "sparse_ffn")


def build_harness(prefetch=2):
    src = f"/* sparse_ffn.c {_sig()} */\n#define KS_PF {prefetch}\n" + _src("bench_sparse.c")
    return _compile(src, (*FLAGS, "-I", os.path.dirname(toolchain.data_path("sparse_ffn.c"))), "bench_sparse", shared=False)


def bench(mode, density=None, thr=None, rank=128, d=2048, dff=6144, threads=8, regime="hot", secs=1.0, prefetch=2, seed=1):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    os.remove(tmp)
    cmd = [build_harness(prefetch), "--mode", mode, "--d", str(d), "--dff", str(dff), "--rank", str(rank), "--threads",
           str(threads), "--regime", regime, "--secs", str(secs), "--seed", str(seed), "--csv", tmp]  # fmt: skip
    if density is not None:
        cmd += ["--density", str(density)]
    if thr is not None:
        cmd += ["--thr", str(thr)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if not os.path.exists(tmp):
        raise RuntimeError(f"bench_sparse failed (exit {r.returncode}): {(r.stderr or r.stdout)[-1000:]}")
    with open(tmp) as fh:
        row = dict(zip(COLUMNS, fh.read().strip().split(",")))
    os.remove(tmp)
    for k in ("density", "us_per_call", "GBps", "relerr", "cpu_s", "calls"):
        row[k] = float(row[k])
    row["cpu_us"] = row["cpu_s"] / row["calls"] * 1e6
    return row
