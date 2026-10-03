"""Host detection, compilers and the build cache.

Environment overrides:
    KURN_CC          C compiler for host targets (default: $CC, then gcc, cc, clang)
    KURN_CROSS_CC    AArch64 cross compiler for `neon` on x86 hosts
                     (default: aarch64-linux-gnu-gcc, then `zig cc -target aarch64-linux-gnu`)
    KURN_QEMU        command prefix to run AArch64 binaries on x86 hosts
                     (default: `qemu-aarch64 -L /usr/aarch64-linux-gnu` when both exist)
    KURN_CACHE_DIR   build cache (default: $XDG_CACHE_HOME/kurn or ~/.cache/kurn)
"""

import functools
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from importlib import resources

from .kernels import generate
from .targets import TARGETS


class BuildError(Exception):
    pass


def host_arch():
    m = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def target_arch(target):
    arch = TARGETS[target].arch
    return host_arch() if arch == "any" else arch


@functools.cache
def cpu_flags():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("flags", "features"):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()


def data_path(name):
    return str(resources.files("kurn") / "data" / name)


def cache_dir():
    d = os.environ.get("KURN_CACHE_DIR") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "kurn")
    os.makedirs(d, exist_ok=True)
    return d


def _which_cmd(cmd):
    parts = shlex.split(cmd)
    return parts if parts and shutil.which(parts[0]) else None


@functools.cache
def host_cc():
    for cand in (os.environ.get("KURN_CC"), os.environ.get("CC"), "gcc", "cc", "clang"):
        if cand and (cmd := _which_cmd(cand)):
            return tuple(cmd)
    raise BuildError("no C compiler found (set KURN_CC)")


@functools.cache
def cross_cc(arch):
    if arch == "aarch64":
        cands = (os.environ.get("KURN_CROSS_CC"), "aarch64-linux-gnu-gcc", "zig cc -target aarch64-linux-gnu")
    elif arch == "x86_64":
        cands = (os.environ.get("KURN_CROSS_CC"), "x86_64-linux-gnu-gcc", "zig cc -target x86_64-linux-gnu")
    else:
        cands = ()
    for cand in cands:
        if cand and (cmd := _which_cmd(cand)):
            return tuple(cmd)
    return None


def cc_for(target):
    arch = target_arch(target)
    if arch == host_arch():
        return host_cc()
    cc = cross_cc(arch)
    if not cc:
        raise BuildError(f"target {target} needs an {arch} compiler; install one or set KURN_CROSS_CC")
    return cc


def flags_for(target):
    if TARGETS[target].arch == "any" and host_arch() != "x86_64":
        return ["-O3"]
    return list(TARGETS[target].flags)


@functools.cache
def qemu(arch="aarch64"):
    env = os.environ.get("KURN_QEMU")
    if env:
        return tuple(shlex.split(env))
    sysroot = f"/usr/{arch}-linux-gnu"
    if shutil.which(f"qemu-{arch}") and os.path.isdir(sysroot):
        return (f"qemu-{arch}", "-L", sysroot)
    return None


def run_mode(target):
    """How a target can be executed here: ("native" | "qemu" | None, reason)."""
    arch = target_arch(target)
    host = host_arch()
    if TARGETS[target].arch == "any" and host != "x86_64":
        return "native", ""
    if arch == host:
        missing = sorted(TARGETS[target].requires - cpu_flags())
        return ("native", "") if not missing else (None, f"host CPU lacks {', '.join(missing)}")
    if arch == "aarch64" and qemu(arch) and cross_cc(arch):
        return "qemu", ""
    return None, f"{arch} target on a {host} host (no qemu-{arch} + cross compiler)"


def _sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


def _compile(args, out):
    tmp = f"{out}.tmp{os.getpid()}"
    r = subprocess.run([*args, "-o", tmp], capture_output=True, text=True)
    if r.returncode:
        raise BuildError(f"C compile failed:\n$ {shlex.join(args)}\n{r.stderr[:4000]}")
    os.replace(tmp, out)


def compile_source(src, target, out_dir=None, stem="kernel", extra_flags=(), shared=True):
    """Compile C source for a target into a cached shared library (or object). Returns its path."""
    cc, flags = cc_for(target), flags_for(target) + list(extra_flags)
    if os.path.basename(cc[0]) == "zig":  # zig cc spells CPU names with underscores (neoverse_n1)
        flags = [f"-mcpu={f[6:].replace('-', '_')}" if f.startswith("-mcpu=") else f for f in flags]
    with open(data_path("kurn.h"), "rb") as fh:
        header = fh.read()
    h = _sha(src, header, shlex.join(cc), shlex.join(flags), shared)
    d = out_dir or os.path.join(cache_dir(), "kernels")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{stem}_{h}")
    out = base + (".so" if shared else ".o")
    if os.path.exists(out):
        return out
    with open(base + ".c", "w") as fh:
        fh.write(src)
    args = [*cc, *flags, "-I", os.path.dirname(data_path("kurn.h"))]
    args += ["-shared", "-fPIC", base + ".c"] if shared else ["-c", base + ".c"]
    _compile(args, out)
    return out


def build(c, out_dir=None, extra_flags=()):
    """Generate and compile a resolved config. Returns the shared library path."""
    stem = f"{c['weights']}_{c['op']}_{c['target']}"
    return compile_source(generate(c), c["target"], out_dir, stem, extra_flags)


def build_harness(arch=None):
    """Compile the bundled benchmark harness for `arch` (default: host). Returns its path."""
    arch = arch or host_arch()
    if arch == host_arch():
        cc = host_cc()
        flags = ["-O3", "-march=native"] if arch == "x86_64" else ["-O3"]
    else:
        cc = cross_cc(arch)
        if not cc:
            raise BuildError(f"no {arch} cross compiler for the harness (set KURN_CROSS_CC)")
        flags = ["-O2"]
    with open(data_path("bench.c"), "rb") as a, open(data_path("kurn.h"), "rb") as b:
        h = _sha(a.read(), b.read(), shlex.join(cc), shlex.join(flags))
    d = os.path.join(cache_dir(), "harness")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"bench_{arch}_{h}")
    if not os.path.exists(out):
        _compile([*cc, *flags, "-I", os.path.dirname(data_path("kurn.h")), data_path("bench.c"), "-lpthread", "-ldl", "-lm"], out)
    return out


def harness_command(target, harness=None):
    """Command prefix that runs the harness for a target's kernels (qemu-wrapped if needed)."""
    mode, why = run_mode(target)
    if mode is None:
        raise BuildError(f"cannot run {target} here: {why}")
    if harness:
        return [harness]
    arch = target_arch(target)
    path = build_harness(arch)
    return [*qemu(arch), path] if mode == "qemu" else [path]


def scratch_dir():
    return tempfile.mkdtemp(prefix="kurn-")
