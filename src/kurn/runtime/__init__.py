"""kurn runtime: a dependency-free C runtime for decode-time parallel loops
(`kurn_rt.h` / `kurn_rt.c`: pinned thread pool, wait primitives, partitioners,
prefetch helpers) plus `rtbench`, a model-shaped decode microbenchmark, and
`rt_selftest`. This module finds, builds (cached) and runs them.

    from kurn import runtime
    runtime.sources()                    # paths to embed in another code base
    so = runtime.default_kernel()        # tuned Q8_0 GEMV (vnni16) shared library
    row = runtime.rtbench(so, workload="olmoe-moe", part="balanced", threads=8)
"""

import csv
import io
import os
import shlex
import subprocess
from importlib import resources

from ..toolchain import _compile, _sha, cache_dir, host_arch, host_cc

FILES = ("kurn_rt.h", "kurn_rt.c")
PARTITIONERS = ("static", "balanced", "streamk", "steal", "ggml")
WAITS = ("spin", "futex", "hybrid:N", "umwait")


def path(name):
    return str(resources.files("kurn") / "runtime" / name)


def sources():
    """Absolute paths of the runtime's header and source (copy them into a host code base)."""
    return [path(f) for f in FILES]


def _flags():
    return ["-O3", "-march=native"] if host_arch() == "x86_64" else ["-O3"]


def _build(main, stem, libs=("-lpthread",)):
    cc, flags = host_cc(), _flags()
    blobs = []
    for f in (main, *FILES):
        with open(path(f), "rb") as fh:
            blobs.append(fh.read())
    out_dir = os.path.join(cache_dir(), "runtime")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{stem}_{_sha(*blobs, shlex.join(cc), shlex.join(flags))}")
    if not os.path.exists(out):
        _compile([*cc, *flags, "-Wall", "-Wextra", path(main), path("kurn_rt.c"), *libs], out)
    return out


def build_rtbench():
    """Compile rtbench (cached by source hash). Returns its path."""
    return _build("rtbench.c", "rtbench", ("-lpthread", "-ldl", "-lm"))


def build_selftest():
    return _build("rt_selftest.c", "rt_selftest")


def default_kernel(target="avx512_vnni", **overrides):
    """Build the tuned Q8_0 decode GEMV (vnni16, packed, rows 8) for `target` and return the .so path.
    For targets without vnni16 the native layout is used (rtbench handles both)."""
    from ..spec import resolve
    from ..toolchain import build

    c = {"op": "gemv", "weights": "q8_0", "target": target}
    if target == "avx512_vnni":
        c.update(layout="vnni16", align="packed", rows=8)
    c.update(overrides)
    return build(resolve(c))


def rtbench(impl, workload="olmoe-moe", part="balanced", threads=8, regime="cold", secs=2.0, extra=(), timeout=900):
    """Run rtbench once; returns the CSV row as a dict (numbers converted where possible)."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="kurn-rt-") as d:
        out = os.path.join(d, "row.csv")
        cmd = [build_rtbench(), "--impl", impl, "--workload", workload, "--part", part, "--threads", str(threads),
               "--regime", regime, "--secs", str(secs), "--csv", out, *map(str, extra)]  # fmt: skip
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if not os.path.exists(out):
            raise RuntimeError(f"rtbench failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[:2000]}")
        with open(out) as fh:
            row = next(csv.DictReader(io.StringIO(fh.read())))
    for k, v in row.items():
        try:
            row[k] = float(v)
        except ValueError:
            pass
    row["stdout"] = r.stdout
    return row
