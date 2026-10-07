"""`op attn` on `target cuda`: flash attention / flash-decoding for A100 (sm_80).

Same semantics as the CPU op (kurn.attention, kurn_attn.h): F16 / BF16 / Q8_0 KV, GQA row packing, MLA (v aliases
k), causal by position and an optional fp16 mask, split-KV with an LSE merge. The ABI is data/kurn_gpu_attn.h and
the kernel template data/attn_kernel.cu.

    op       attn
    target   cuda
    kv       q8_0          # f16 | bf16 | q8_0
    dk       128           # head dims (dv defaults to dk; MLA: dk 576, dv 512, mla 1)
    tk       64            # KV tokens per tile
    wm       1             # warps along rows (16 GQA-packed rows each)
    wn       4             # warps per row group: split the tile for QK^T and the output columns for PV
    split    0             # KV splits per (row tile, kv head); 0 = one wave of resident CTAs, >= max(tk, 64) tokens each
    qsplit   1             # bf16 / fp8: q as hi + lo tiles (two QK^T MMAs; bf16 ~16 bits of q instead of 8)
    deq      1             # q8_0: dequantize MMA fragments in registers (0: shared-memory pass into f16 tiles)
    merge    0             # 1: the last split CTA merges (no merge launch; one call of a library at a time per device)
    heads 32  kv_heads 8  nq 1  nkv 8192          # problem (not codegen)
    tune     tk=32,64,128 wn=1,2,4 split=0,8,16

Numerics: q is pre-scaled into the log2 domain and rounded to the MMA type with K and P (f16 for F16 and Q8_0 KV,
bf16 for BF16 KV; bf16 q as hi + lo by default), with f32 accumulation and softmax. Q8_0 values d * q are rounded to
f16 once (identically in registers and in shared memory).
"""

import itertools
import json
import os
import re
import shlex
import subprocess

from ..spec import SpecError
from ..spec import parse as _parse
from .toolchain import (
    MIN_NVCC,
    GpuBuildError,
    _out_dir,
    _run,
    _sha,
    _warn,
    arch_flags,
    cant_build,
    cxx,
    cxx_is_clang,
    cxx_problem,
    data_path,
    nvcc,
    nvcc_name,
    nvcc_supports,
    nvcc_version,
    parse_ptxas,
)

KV_FORMATS = {"f16": 0, "bf16": 1, "q8_0": 2, "fp8": 3}  # fp8 = e4m3, GPU only: needs FP8 mma.sync (sm_89+)
FP8_ARCHS = ("sm_100", "sm_120")
HEAD_DIMS = {64: (64,), 128: (128,), 256: (256,), 576: (512,)}  # dk -> legal dv (as the CPU op)
ARCH = "sm_80"  # the default and measured target (A100)
# Tiers a config is validated and tuned for: sm_80 (A100, measured), sm_100 (B200 / GB200) and sm_120 (RTX 50), the
# last two compile-only. Every build carries SASS for each of them (when nvcc can target it) plus compute_80 PTX.
ARCHS = ("sm_80", "sm_100", "sm_120")
FATBIN_ARCHS = ARCHS
SMEM_LIMITS = {"sm_80": 166912, "sm_100": 232448, "sm_120": 101376}  # opt-in dynamic shared memory per block
SMEM_MAX = SMEM_LIMITS[ARCH]
# Default tiles per tier and head dim: (tk, wn). sm_120 has 99 KB per block, so dk 256 drops to 32-token tiles; sm_100
# has 227 KB, so MLA keeps 64-token tiles. MLA everywhere uses 4 warps: O (dv 512) is 64 registers per thread instead
# of 128 with wn 2 (A100 run 1: tk 32 / wn 2 sat near 240 registers with 64 threads per CTA and ran at ~200 GB/s); on
# sm_80 / sm_120 that means 8-token QK^T warp slices (tk 32 / wn 4).
ARCH_DEFAULTS = {
    "sm_80": {64: (64, 4), 128: (64, 4), 256: (64, 4), 576: (32, 4)},
    "sm_100": {64: (64, 4), 128: (64, 4), 256: (64, 4), 576: (64, 4)},
    "sm_120": {64: (64, 4), 128: (64, 4), 256: (32, 2), 576: (32, 4)},
}
# Other GPUs map to the tier with their per-block shared memory (sm_86/89: 99 KB like sm_120; sm_90/103: 227 KB).
TIER_OF = {"sm_80": "sm_80", "sm_86": "sm_120", "sm_87": "sm_80", "sm_89": "sm_120", "sm_90": "sm_100", "sm_100": "sm_100",
           "sm_101": "sm_100", "sm_103": "sm_100", "sm_110": "sm_100", "sm_120": "sm_120", "sm_121": "sm_120"}  # fmt: skip
REG_BUDGET = 200  # estimated registers per thread above which ptxas would likely spill (255 cap, launch bounds)
STAGES = 2

SCHEDULE = {
    "kv": lambda c: tuple(k for k in KV_FORMATS if k != "fp8" or c["arch"] in FP8_ARCHS),
    "dk": lambda c: tuple(HEAD_DIMS),
    "dv": lambda c: HEAD_DIMS[c["dk"]],
    "mla": lambda c: (0, 1) if c["dv"] <= c["dk"] else (0,),
    "tk": lambda c: (32, 64, 128),
    "wm": lambda c: (1, 2, 4),
    "wn": lambda c: (1, 2, 4),
    "split": lambda c: (0, 1, 2, 4, 8, 16, 32, 64),
    "qsplit": lambda c: (1, 0) if c["kv"] in ("fp8", "bf16") else (0,),
    "deq": lambda c: (1, 0) if c["kv"] == "q8_0" else (0,),
    "merge": lambda c: (0, 1),
}
DEFAULTS = {"kv": "f16", "dk": 128, "tk": 64, "wm": 1, "wn": 4, "split": 0}
CODEGEN_KEYS = ("arch", "kv", "dk", "dv", "mla", "tk", "wm", "wn", "split", "qsplit", "deq", "merge")
PROBLEM = {"heads": 32, "kv_heads": 8, "nq": 1, "nkv": 4096, "pos0": -1, "causal": 1, "mask": 0, "layout": 0}
KNOWN = ("kernel", "op", "target", "arch") + tuple(SCHEDULE) + tuple(PROBLEM)
# max |out - ref| / max |ref| against the float64 reference on the stored (dequantized) K/V. f16: q, k, P rounded to
# 11 bits; bf16: 8 bits (the CPU AMX-BF16 engine's tolerance); Q8_0 adds one f16 rounding of d * q.
TOL = {"f16": 4e-3, "bf16": 6e-3, "bf16_q1": 1.5e-2, "q8_0": 4e-3, "fp8": 1.5e-2, "fp8_q1": 1e-1}
# bf16: qsplit 1 (default) keeps q as hi + lo bf16 (~16 bits), which leaves P's bf16 rounding (~3e-3 worst on the
# awkward shapes); qsplit 0 rounds q to 8 bits, whose score error the peaked test distribution amplifies (A100 run 1:
# 1.4e-2 on Qwen3-1.7B at 16K, 1.3e-2 on the emulator; 1.8e-3 with qsplit 1).
# fp8: K and V are exact (the reference reads the stored e4m3) but q is rounded to e4m3 for the FP8 QK^T MMA. qsplit 1
# (default) keeps q as hi + lo e4m3 (about 8 bits, like bf16); qsplit 0 keeps 3 bits, which on the deliberately peaked
# test distribution (q ~ N(0, 3^2)) moves the outputs by up to ~8%. Independently of q's rounding, every fp8 run is
# also checked against a reference with q rounded exactly as the kernel does (TOL_Q8): that isolates the kernel's own
# error (P in f16, f32 accumulation order).
TOL_Q8 = 4e-3


def tol(c):
    return TOL[c["kv"] + "_q1"] if c["kv"] in ("fp8", "bf16") and not c["qsplit"] else TOL[c["kv"]]


def _row_bytes(kv, d):
    return d // 32 * 34 if kv == "q8_0" else d if kv == "fp8" else 2 * d


def smem_bytes(c):
    al = lambda x: (x + 15) // 16 * 16  # noqa: E731
    bm, tk, dk, dv = 16 * c["wm"], c["tk"], c["dk"], c["dv"]
    if c["kv"] == "fp8":  # e4m3 q + row scales, raw e4m3 K (and V) stages, V converted to f16, P, reductions
        q = (1 + c["qsplit"]) * al(bm * (dk + 16)) + al(bm * 4)
        raw = al(tk * (dk + 16)) + (0 if c["mla"] else al(tk * (dv + 16)))
        return q + STAGES * raw + al(tk * (dv + 8) * 2) + al(bm * (tk + 8) * 2) + al(c["wn"] * bm * 4)
    kbuf = (0 if c.get("deq", 1) else 1) if c["kv"] == "q8_0" else STAGES
    q = (1 + c["qsplit"]) * al(bm * (dk + 8) * 2)
    k = al(tk * (dk + 8) * 2)
    v = 0 if c["mla"] else al(tk * (dv + 8) * 2)
    raw = al(tk * (_row_bytes(c["kv"], dk) + (0 if c["mla"] else _row_bytes(c["kv"], dv)))) if c["kv"] == "q8_0" else 0
    p = al(bm * (tk + 8) * 2)
    red = al(c["wn"] * bm * 4)
    return q + kbuf * (k + v) + STAGES * raw + p + red


def est_regs(c):
    """Accumulators: S (tk/wn/8 n-tiles x 4) and O (dv/wn/8 x 4) per thread, plus addressing and softmax state; Q8_0
    register dequant holds raw bytes, scales and row pointers on top (MLA tk 64 / wn 2, estimated 200, spilled at 255)."""
    deq = 16 if c["kv"] == "q8_0" and c.get("deq", 1) else 0
    return c["tk"] // c["wn"] // 2 + c["dv"] // c["wn"] // 2 + 56 + deq


INVALID = [
    (lambda c: (c["tk"] // c["wn"]) % 8, "tk / wn must be a multiple of 8 (QK^T n-tiles are 8 tokens)"),
    (lambda c: (c["dv"] // c["wn"]) % 16, "dv / wn must be a multiple of 16 (PV n-tiles are loaded in pairs)"),
    (lambda c: c["wm"] * c["wn"] > 16, "at most 16 warps per CTA"),
    (
        lambda c: smem_bytes(c) > SMEM_LIMITS[c["arch"]],
        "shared memory exceeds the arch's per-block limit (sm_80 163 KB, sm_100 227 KB, sm_120 99 KB): smaller tk, or q8_0 / mla",
    ),
    (lambda c: est_regs(c) > REG_BUDGET, "accumulators need too many registers (use more wn warps or a smaller tk)"),
    (
        lambda c: est_regs(c) > 0.9 * min(255, 65536 // (32 * c["wm"] * c["wn"])),
        "too many registers for the launch bounds (512 threads cap a thread at 128): use fewer wm x wn warps",
    ),
]


def parse(text):
    return _parse(text)


def load(path):
    with open(path) as fh:
        return parse(fh.read())


def resolve(spec, overrides=None):
    """Validate an `op attn` cuda spec (plus overrides) and fill defaults. Returns the config dict."""
    c = {**spec, **(overrides or {})}
    if c.get("op", "attn") != "attn":
        raise SpecError(f"op {c['op']!r}: kurn.gpu.attn handles op attn only")
    c["op"] = "attn"
    if c.get("target", "cuda") != "cuda":
        raise SpecError(f"target {c['target']!r}: kurn.gpu.attn handles target cuda only")
    c["target"] = "cuda"
    if c.setdefault("arch", ARCH) not in ARCHS:
        raise SpecError(f"arch {c['arch']!r}: GPU attention tiers are {list(ARCHS)} (map other GPUs with tier_for)")
    if c.get("kv") == "fp8" and c["arch"] not in FP8_ARCHS:
        raise SpecError(f"kv fp8 runs QK^T on FP8 mma.sync (sm_89+): use arch {' or '.join(FP8_ARCHS)}, not {c['arch']}")
    for k in c:
        if k not in KNOWN:
            raise SpecError(f"unknown key {k!r} for op attn target cuda: expected one of {sorted(KNOWN)}")
    for k, legal in SCHEDULE.items():
        allowed = legal(c)
        if k not in c:
            if k == "mla":
                c[k] = 1 if c["dk"] == 576 else 0
            elif k in ("tk", "wn"):
                c[k] = ARCH_DEFAULTS[c["arch"]][c["dk"]][0 if k == "tk" else 1]
            elif k == "qsplit" and c["kv"] == "bf16":  # the lo q tile only where it fits (not bf16 MLA on sm_120)
                c[k] = 1 if smem_bytes({**c, k: 1, "deq": 0}) <= SMEM_LIMITS[c["arch"]] else 0
            else:
                c[k] = DEFAULTS[k] if DEFAULTS.get(k) in allowed else allowed[0]
        if c[k] not in allowed:
            raise SpecError(f"{k}={c[k]!r} not allowed for attn/cuda: expected one of {list(allowed)}")
    for k, v in PROBLEM.items():
        c.setdefault(k, v)
        if not isinstance(c[k], int):
            raise SpecError(f"{k}={c[k]!r}: expected an integer")
    if c["heads"] % c["kv_heads"]:
        raise SpecError(f"heads={c['heads']} must be a multiple of kv_heads={c['kv_heads']}")
    for bad, msg in INVALID:
        if bad(c):
            raise SpecError(msg)
    c.setdefault("kernel", f"attn_{c['kv']}_d{c['dk']}_cuda")
    return c


def tier_for(arch):
    """The attention tier (sm_80 / sm_100 / sm_120) whose limits fit GPU `arch` (e.g. 'sm_86' -> 'sm_120')."""
    if arch in TIER_OF:
        return TIER_OF[arch]
    major = int(arch.split("_")[1]) // 10 if arch and arch.startswith("sm_") else 8
    return "sm_120" if major >= 12 else "sm_100" if major >= 9 else "sm_80"


def kernel_for(kv, dk, arch=None, **problem):
    """Per-arch dispatch: the default config for KV format `kv` and head dim `dk` on GPU `arch` (default: the local
    GPU, else sm_80). MLA (dk 576) picks mla 1. Extra keys (heads, kv_heads, nq, nkv, ...) set the problem."""
    if arch is None:
        from .harness import detect_arch

        arch = detect_arch() or ARCH
    return resolve({"kv": kv, "dk": dk, "arch": tier_for(arch), **problem})


def label(c):
    return " ".join(f"{k}={c[k]}" for k in CODEGEN_KEYS)


def config_key(c):
    return tuple(c[k] for k in CODEGEN_KEYS)


def generate(c):
    """Resolved config -> CUDA C++ implementing kurn_gpu_attn.h."""
    from .codegen import CPASYNC, PRELUDE
    from .mma import HELPERS

    with open(data_path("attn_kernel.cu")) as fh:
        body = fh.read()
    defs = {"KGA_DK": c["dk"], "KGA_DV": c["dv"], "KGA_KV": KV_FORMATS[c["kv"]], "KGA_TK": c["tk"], "KGA_WM": c["wm"],
            "KGA_WN": c["wn"], "KGA_STAGES": STAGES, "KGA_SPLIT": c["split"], "KGA_MLA": c["mla"],
            "KGA_QSPLIT": c["qsplit"], "KGA_DEQ": c["deq"], "KGA_MERGE": c["merge"]}  # fmt: skip
    head = [f"// kurn GPU attention kernel (tuned for {c['arch']}): {label(c)}"]
    head += [f"#define {k} {v}" for k, v in defs.items()]
    head.append(f'#define KGA_CONFIG "op=attn {label(c)}"')
    return "\n".join(head) + "\n" + PRELUDE + CPASYNC + HELPERS + body


# --------------------------------------------------------------------------- builds


def emu_build(c, src=None):
    """Compile with the CPU warp emulator into an `emu_attn` executable."""
    problem = cxx_problem()
    if problem:
        from .toolchain import GpuToolchainError

        raise GpuToolchainError(problem)
    src = src or generate(c)
    flags = ["-std=c++17", "-O1", "-g0", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-Wno-unused-parameter",
             "-Wno-unused-function", "-Wno-unused-variable", "-Werror", "-DKURN_EMU", f"-DKGA_MLA_BUILD={c['mla']}"]  # fmt: skip
    if cxx_is_clang():
        flags.append("-Wno-pass-failed")
    deps = b"".join(open(data_path(n), "rb").read() for n in
                    ("kurn_cuemu.h", "kurn_gpu_attn.h", "kurn_gpu_attn_ref.h", "kurn_gpu_ref.h", "emu_attn_main.cpp"))  # fmt: skip
    h = _sha(src, deps, shlex.join(cxx()), shlex.join(flags))
    d = _out_dir("emu")
    exe = os.path.join(d, f"attn_{c['kv']}_d{c['dk']}_{h}")
    if os.path.exists(exe):
        return exe
    cu = exe + ".cu"
    with open(cu, "w") as fh:
        fh.write(src)
    tmp = f"{exe}.tmp{os.getpid()}"
    _run([*cxx(), *flags, "-I", os.path.dirname(data_path("kurn_gpu_attn.h")), "-x", "c++", cu, "-x", "c++",
          data_path("emu_attn_main.cpp"), "-o", tmp], "emulator build")  # fmt: skip
    os.replace(tmp, exe)
    return exe


def _requested_archs():
    """An explicit arch list (`--archs` on the CLI or KURN_GPU_ARCHS='sm_80,sm_120'), or None for the defaults."""
    env = os.environ.get("KURN_GPU_ARCHS", "").strip()
    if not env:
        return None
    archs = [x.strip() for x in env.split(",") if x.strip()]
    bad = [x for x in archs if not re.fullmatch(r"sm_\d+[af]?", x)]
    if bad:
        raise SpecError(f"KURN_GPU_ARCHS / --archs: {bad} are not arch names (expected e.g. sm_80,sm_100,sm_120)")
    return archs


def target_archs(c=None):
    """The archs a build embeds SASS for, decided by the nvcc that actually builds (on the kit's target box that is the
    box's own nvcc) - never by where kurn generated the source.
    - Explicit request (--archs / KURN_GPU_ARCHS): exactly those, filtered only by the config's features (fp8 needs
      sm_89+); an arch this nvcc can't build is an error, not a silent drop.
    - Default: the three tiers; ones this nvcc can't build are skipped with a warning naming them. Those GPUs still
      run the kernel by JIT from the embedded PTX (see ptx_arch)."""
    need = FP8_ARCH_MIN if c is not None and c["kv"] == "fp8" else 0
    req = _requested_archs()
    if req is not None:
        out = [a for a in req if int(re.sub(r"\D", "", a.split("_")[1])) >= need]
        if c is not None and len(out) < len(req):
            _warn(f"kv fp8 needs FP8 mma.sync (sm_89+): not building {sorted(set(req) - set(out))} for this config")
        missing = [a for a in out if nvcc_supports(a) is False]
        if missing:
            raise GpuBuildError(f"requested archs {missing}: {'; '.join(cant_build(a) for a in missing)}")
        return out
    want = [a for a in FATBIN_ARCHS if int(a.split("_")[1]) >= need]
    out = [a for a in want if nvcc_supports(a) is not False]
    if len(out) < len(want):
        skipped = [a for a in want if a not in out]
        ptx = ptx_arch(c, out)
        jit = f"; those GPUs JIT the embedded compute_{ptx.split('_')[1]} PTX if their driver supports it" if ptx else ""
        need = max((MIN_NVCC[a] for a in skipped if a in MIN_NVCC), default=None)
        need = f" (needs CUDA {need[0]}.{need[1]}+)" if need else ""
        _warn(f"{nvcc_name()} can't build {', '.join(skipped)} SASS{need}: skipping {'it' if len(skipped) == 1 else 'them'}"
              f"{jit} (KURN_GPU_ARCHS / --archs makes missing archs an error)")  # fmt: skip
    return out


FP8_ARCH_MIN = 89  # FP8 mma.sync


def ptx_arch(c=None, sass=None):
    """The arch whose PTX the fatbin embeds: the lowest SASS arch, or - when no SASS arch can be built (e.g. fp8 with
    nvcc < 12.8) - the newest feature-capable arch this nvcc can build PTX for (fp8: compute_89/90)."""
    sass = target_archs(c) if sass is None else sass
    if sass:
        return sass[0]
    need = FP8_ARCH_MIN if c is not None and c["kv"] == "fp8" else 80
    cands = [a for a in ("sm_80", "sm_89", "sm_90") if int(a.split("_")[1]) >= need and nvcc_supports(a) is not False]
    return cands[-1] if cands else None


def fatbin_archs(c=None):
    """SASS archs of a build (alias of target_archs)."""
    return target_archs(c)


def fatbin_flags(c=None):
    """SASS for every target_archs(c) entry plus PTX of ptx_arch(c), so drivers can JIT the kernel for later GPUs."""
    archs = target_archs(c)
    ptx = ptx_arch(c, archs)
    if ptx is None:
        why = " (kv fp8 needs FP8 mma.sync: sm_89+, CUDA 11.8+)" if c is not None and c["kv"] == "fp8" else ""
        raise GpuBuildError(f"{nvcc_name()} can't build any arch this config needs{why}")
    out = []
    for a in archs:
        n = a.split("_")[1]
        out += ["-gencode", f"arch=compute_{n},code=" + (f"[sm_{n},compute_{n}]" if a == ptx else f"sm_{n}")]
    if ptx not in archs:
        n = ptx.split("_")[1]
        out += ["-gencode", f"arch=compute_{n},code=compute_{n}"]
    return out


def run_mode(arch, c=None):
    """How GPU `arch` runs a library built here: 'native SASS', 'JIT from compute_XX PTX' or 'cannot run (...)'."""
    archs = target_archs(c)
    if arch in archs:
        return "native SASS"
    ptx = ptx_arch(c, archs)
    if ptx and int(re.sub(r"\D", "", arch.split("_")[1])) >= int(ptx.split("_")[1]):
        return f"JIT from compute_{ptx.split('_')[1]} PTX"
    return f"cannot run (no SASS or PTX at or below {arch})"


def nvcc_build(c, src=None, out_dir=None):
    """nvcc -> shared library implementing kurn_gpu_attn.h: a fatbin with sm_80 and sm_120 SASS and compute_80 PTX.
    Returns (path, ptxas resources keyed 'arch:kernel')."""
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found (install the CUDA toolkit or set KURN_NVCC)")
    from .cudaenv import build_flags

    src = src or generate(c)
    flags = ["-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-lineinfo", *fatbin_flags(c),
             "-I", os.path.dirname(data_path("kurn_gpu_attn.h")), *build_flags()]  # fmt: skip
    h = _sha(src, open(data_path("kurn_gpu_attn.h"), "rb").read(), n, nvcc_version(), shlex.join(flags))
    d = out_dir or _out_dir("cuda")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"attn_{c['kv']}_d{c['dk']}_{h}.so")
    rep = so + ".ptxas.txt"
    if not (os.path.exists(so) and os.path.exists(rep)):
        cu = so[:-3] + ".cu"
        with open(cu, "w") as fh:
            fh.write(src)
        tmp = f"{so}.tmp{os.getpid()}"
        r = _run([n, *flags, cu, "-o", tmp], "nvcc build")
        with open(rep, "w") as fh:
            fh.write(r.stderr)
        os.replace(tmp, so)
    with open(rep) as fh:
        return so, parse_ptxas(fh.read())


def ptxas(c):
    """{arch: {kernel: {regs, spill}}} for each fatbin arch. (Static shared memory is 0: the tile is dynamic, see
    smem_bytes; the per-arch launch limit is SMEM_LIMITS.)"""
    _, rep = nvcc_build(c)
    out = {}
    for key, r in rep.items():
        arch, _, k = key.partition(":")
        name = "kga_main" if "kga_main" in k else "kga_merge" if "kga_merge" in k else k
        out.setdefault(arch, {})[name] = {"regs": r["regs"], "spill": r["spill_st"] + r["spill_ld"] + r["stack"]}
    return out


def fatbin_contents(so):
    """cuobjdump listing of a built library: (SASS archs, PTX archs)."""
    from .cudaenv import cuobjdump

    exe = cuobjdump()
    elf = subprocess.run([exe, "--list-elf", so], capture_output=True, text=True).stdout
    ptx = subprocess.run([exe, "--list-ptx", so], capture_output=True, text=True).stdout
    grab = lambda txt, kind: sorted(set(re.findall(rf"\.({kind}_\d+)\.", txt)))  # noqa: E731
    return grab(elf, "sm"), grab(ptx, "sm")


A100_SMS = 108
A100_SMEM_SM = 167936  # shared memory per SM
MAX_SPLIT = 128  # KGA_MAX_SPLIT


def resident(c, smem_sm=A100_SMEM_SM):
    """CTAs per SM by shared memory, capped at 2 (kga_resident)."""
    return max(1, min(2, smem_sm // (smem_bytes(c) + 1024)))


def splits_for(c, shape=None, sms=A100_SMS, smem_sm=A100_SMEM_SM):
    """KV splits kga_run picks for problem `shape` on a GPU with `sms` SMs (mirror of kga_make_plan): one wave of
    resident CTAs, each split at least max(tk, 64) tokens."""
    sh = {**PROBLEM, **{k: c[k] for k in PROBLEM if k in c}, **(shape or {})}
    kvh = 1 if c["mla"] else sh["kv_heads"]
    ctas = _cdiv(sh["nq"] * (sh["heads"] // kvh), 16 * c["wm"]) * kvh
    ns = c["split"]
    if ns <= 0:
        want = resident(c, smem_sm) * sms
        ns = _cdiv(want, ctas) if ctas < want else 1
        ns = min(ns, max(1, sh["nkv"] // max(c["tk"], 64)))
    ns = max(1, min(ns, MAX_SPLIT))
    chunk = _cdiv(_cdiv(sh["nkv"], ns), c["tk"]) * c["tk"]
    return max(1, _cdiv(sh["nkv"], chunk))


def _cdiv(a, b):
    return (a + b - 1) // b


def archs_for(c):
    """Tiers this config can run on: its per-block shared memory fits (kga_run returns -4 elsewhere) and the tier has
    the features it needs (fp8: FP8 mma.sync). A property of the config only - independent of the local nvcc, which
    decides what a particular build embeds (target_archs)."""
    feat = FP8_ARCHS if c["kv"] == "fp8" else FATBIN_ARCHS
    return [a for a in FATBIN_ARCHS if smem_bytes(c) <= SMEM_LIMITS[a] and a in feat]


# --------------------------------------------------------------------------- emulator verification

# The CPU op's awkward shapes (kurn.attention.CHECK_SHAPES: query-tile tails, GQA 1/2/4, causal and mask, decode with
# auto and forced splits, KV-tile tails), plus decode with GQA 8, head-major caches, a fully masked prefix
# (pos0 < 0 rows see nothing) and splits that end inside a tile.
CHECK_SHAPES = (
    {"nq": 1, "nkv": 1000, "heads": 8, "kv_heads": 2},
    {"nq": 3, "nkv": 777, "heads": 4, "kv_heads": 4},
    {"nq": 1, "nkv": 4100, "heads": 16, "kv_heads": 8},
    {"nq": 100, "nkv": 100, "heads": 4, "kv_heads": 2},
    {"nq": 37, "nkv": 300, "heads": 6, "kv_heads": 3},
    {"nq": 70, "nkv": 333, "heads": 8, "kv_heads": 2, "mask": 1},
    {"nq": 64, "nkv": 200, "heads": 2, "kv_heads": 1, "causal": 0},
    {"nq": 129, "nkv": 1500, "heads": 4, "kv_heads": 1},
    {"nq": 1, "nkv": 2049, "heads": 32, "kv_heads": 4, "layout": 1},
    {"nq": 5, "nkv": 64, "heads": 4, "kv_heads": 2, "pos0": -3},
)
QUICK_SHAPES = (CHECK_SHAPES[0], CHECK_SHAPES[4], CHECK_SHAPES[5], CHECK_SHAPES[8], CHECK_SHAPES[9])
MLA_HEADS = 16  # MLA problems: one latent kv head shared by MLA_HEADS query heads


class EmuError(Exception):
    pass


def emu_run(c, shape, seed=1, sched=0, exe=None, cpu_lib=None, sms=None, runs=1):
    """Run one problem on the emulator. Returns {relerr, splits, ok, ...}; with `cpu_lib` (a kurn.attention library)
    also vs_cpu (max |gpu - cpu| / max |cpu| on the same arguments) and cpu_relerr. `sms`: the SM count auto splits
    see (default 8); `runs`: calls on one workspace, the last checked."""
    sh = {**PROBLEM, **shape}
    if c["mla"]:
        sh.update(kv_heads=1, heads=max(sh["heads"], 1))
    exe = exe or emu_build(c)
    args = [str(x) for x in (sh["nq"], sh["nkv"], sh["heads"], sh["kv_heads"], sh["causal"], sh["pos0"], sh["mask"], sh["layout"], seed)]
    env = dict(os.environ)
    if sched:
        env["KEMU_SEED"] = str(sched)
    if cpu_lib:
        env["KGA_CPU_LIB"] = cpu_lib
    if sms:
        env["KEMU_SMS"] = str(sms)
    if runs > 1:
        env["KGA_RUNS"] = str(runs)
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=1800, env=env)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
    if r.returncode or not line:
        raise EmuError(f"emulator run failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-2000:]}")
    res = json.loads(re.sub(r"(?<![\w.])-?(inf|nan)\b", "1e999", line))  # unwritten / non-finite outputs: relerr inf
    if "error" in res:
        raise EmuError(res["error"])
    res["ok"] = res["relerr"] <= tol(c) and res.get("relerr_q8", 0.0) <= TOL_Q8
    return res


def emu_check(c, shapes=CHECK_SHAPES, scheds=(0,), log=None, exe=None):
    """Every shape (and schedule) on the emulator. Returns the worst result (ok False if any failed)."""
    exe = exe or emu_build(c)
    worst = None
    for i, sh in enumerate(shapes):
        if c["mla"]:
            sh = {**sh, "heads": MLA_HEADS if sh["heads"] >= 4 else sh["heads"], "kv_heads": 1}
        for sc in scheds:
            r = emu_run(c, sh, seed=1 + i, sched=sc, exe=exe)
            r["shape"] = sh
            if log:
                log(f"  {sh} sched={sc}: relerr {r['relerr']:.2e} splits {r['splits']} {'ok' if r['ok'] else 'FAIL'}")
            if worst is None or (not r["ok"] and worst["ok"]) or (r["ok"] == worst["ok"] and r["relerr"] > worst["relerr"]):
                worst = r
    return worst


def covering_configs(kv=None, dk=None, arch=ARCH):
    """For tier `arch`: the defaults per (kv, head dims) plus every legal value of every schedule key varied one at a
    time."""
    out, seen = [], set()

    def add(c):
        if config_key(c) not in seen:
            seen.add(config_key(c))
            out.append(c)

    for f, d in itertools.product(KV_FORMATS, HEAD_DIMS):
        if (kv and f != kv) or (dk and d != dk):
            continue
        base = {"op": "attn", "target": "cuda", "kv": f, "dk": d, "arch": arch}
        try:
            dflt = resolve(base)
        except SpecError:
            continue
        add(dflt)
        for k in ("tk", "wm", "wn", "split", "mla"):
            for v in SCHEDULE[k](dflt):
                try:
                    add(resolve(base, {k: v}))
                except SpecError:
                    pass
    return out


def legal_configs(kv=None, dk=None, arch=ARCH):
    for f, d in itertools.product(KV_FORMATS, HEAD_DIMS):
        if (kv and f != kv) or (dk and d != dk):
            continue
        base = {"op": "attn", "target": "cuda", "kv": f, "dk": d, "arch": arch}
        for combo in itertools.product(*(SCHEDULE[k]({"dk": d, "dv": HEAD_DIMS[d][0]}) for k in ("tk", "wm", "wn", "mla"))):
            try:
                yield resolve(base, dict(zip(("tk", "wm", "wn", "mla"), combo)))
            except SpecError:
                continue


# --------------------------------------------------------------------------- GPU harness (A100; RTX 50 compile-only)


class HarnessError(Exception):
    pass


def build_harness(arch=ARCH, out_dir=None, explicit=False):
    """nvcc data/bench_gpu_attn.cu for `arch`. Returns the executable path. When this nvcc can't build `arch` (e.g. an
    RTX 50 box with CUDA 12.4) the harness gets PTX of an older arch and a warning, unless the arch was `explicit`ly
    requested, which is then an error."""
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found (install the CUDA toolkit or set KURN_NVCC)")
    names = ("bench_gpu_attn.cu", "kurn_gpu_attn.h", "kurn_gpu_attn_ref.h", "kurn_gpu_ref.h")
    src = b"".join(open(data_path(x), "rb").read() for x in names)
    from .cudaenv import build_flags

    args = [n, "-O3", "-std=c++17", *arch_flags([arch], fallback=not explicit), "-I", os.path.dirname(data_path("kurn_gpu_attn.h")),
            *build_flags()]  # fmt: skip
    h = _sha(src, n, nvcc_version(), " ".join(args))
    d = out_dir or _out_dir("harness")
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(d, f"bench_gpu_attn_{arch}_{h}")
    if not os.path.exists(exe):
        tmp = f"{exe}.tmp{os.getpid()}"
        _run([*args, data_path("bench_gpu_attn.cu"), "-o", tmp, "-ldl"], "attention harness build")
        os.replace(tmp, exe)
    return exe


def gpu_run(harness, lib, c, shape=None, secs=0.0, reps=3, cold_bytes=3e8, seed=1, check_toks=16, timeout=900):
    """Run one problem on the local GPU. Returns (check row, sample rows)."""
    sh = {**PROBLEM, **{k: c[k] for k in PROBLEM if k in c}, **(shape or {})}
    cmd = [harness, "run", "--lib", lib, "--nq", sh["nq"], "--nkv", sh["nkv"], "--heads", sh["heads"], "--kv-heads", sh["kv_heads"],
           "--causal", sh["causal"], "--pos0", sh["pos0"], "--mask", sh["mask"], "--layout", sh["layout"], "--seed", seed,
           "--tol", tol(c), "--tol-q8", TOL_Q8, "--secs", secs, "--reps", reps, "--cold-bytes", cold_bytes,
           "--check-toks", check_toks]  # fmt: skip
    r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=timeout)
    rows = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    err = next((x for x in rows if x.get("kind") == "error"), None)
    if err or not rows:
        raise HarnessError((err or {}).get("error") or r.stderr.strip()[-2000:] or f"exit {r.returncode}")
    return next(x for x in rows if x["kind"] == "check"), [x for x in rows if x["kind"] == "sample"]


def gpu_check(harness, c, shapes=CHECK_SHAPES, log=None):
    """The emulator's awkward shapes, on the GPU. Returns the worst check row."""
    lib = nvcc_build(c)[0]
    worst = None
    for i, sh in enumerate(shapes):
        if c["mla"]:
            sh = {**sh, "heads": MLA_HEADS if sh["heads"] >= 4 else sh["heads"], "kv_heads": 1}
        chk, _ = gpu_run(harness, lib, c, sh, seed=1 + i, check_toks=64)
        chk["shape"] = sh
        if log:
            log(f"  {sh}: relerr {chk['relerr']:.2e} splits {chk['splits']} {chk['status']}")
        if (
            worst is None
            or (chk["status"] != "ok" and worst["status"] == "ok")
            or ((chk["status"] == "ok") == (worst["status"] == "ok") and chk["relerr"] > worst["relerr"])
        ):
            worst = chk
    return worst


def build_ggml_bench(llama_dir, out_dir=None):
    """nvcc data/bench_ggml_attn.cpp against a llama.cpp checkout built with -DGGML_CUDA=ON (ggml/include, build*/bin).
    Returns the executable path."""
    import glob

    from .cudaenv import build_flags
    from .harness import ggml_paths

    gp = ggml_paths(llama_dir)
    if not gp:
        raise GpuBuildError(f"{llama_dir}: no ggml/include or build*/bin/libggml-cuda.so (build ggml with -DGGML_CUDA=ON)")
    incs, libdir = gp
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found (install the CUDA toolkit or set KURN_NVCC)")
    names = ("bench_ggml_attn.cpp", "kurn_gpu_attn.h", "kurn_gpu_attn_ref.h", "kurn_gpu_ref.h")
    src = b"".join(open(data_path(x), "rb").read() for x in names)
    args = [n, "-O2", "-std=c++17", "-Wno-deprecated-gpu-targets", "-I", os.path.dirname(data_path("kurn_gpu_attn.h")),
            *[f"-I{i}" for i in incs], *build_flags()]  # fmt: skip
    libs = [f"-l{os.path.basename(p)[3:-3]}" for p in sorted(glob.glob(os.path.join(libdir, "libggml*.so")))]
    # libggml-cuda needs libcuda.so.1, which only a box with the driver has (the build can run without one)
    link = ["-L", libdir, *libs, f"-Xlinker=-rpath={libdir}", "-Xlinker=--allow-shlib-undefined"]
    h = _sha(src, n, nvcc_version(), " ".join(args + link), open(os.path.join(incs[0], "ggml.h"), "rb").read())
    d = out_dir or _out_dir("harness")
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(d, f"bench_ggml_attn_{h}")
    if not os.path.exists(exe):
        tmp = f"{exe}.tmp{os.getpid()}"
        _run([*args, "-x", "c++", data_path("bench_ggml_attn.cpp"), "-o", tmp, *link], "ggml attention bench build")
        os.replace(tmp, exe)
    return exe


def ggml_run(exe, c, shape=None, secs=0.0, reps=3, cold_bytes=3e8, seed=1, check_toks=16, timeout=900):
    """llama.cpp's CUDA flash attention on the problem gpu_run would run for config `c` (its kv, dk, dv, mla; the
    tiling keys don't apply). Returns (check row, sample rows)."""
    sh = {**PROBLEM, **{k: c[k] for k in PROBLEM if k in c}, **(shape or {})}
    t = tol({**c, "qsplit": 0}) if c["kv"] == "bf16" else tol(c)
    cmd = [exe, "run", "--kv", c["kv"], "--dk", c["dk"], "--dv", c["dv"], "--mla", c["mla"], "--nq", sh["nq"], "--nkv", sh["nkv"],
           "--heads", sh["heads"], "--kv-heads", sh["kv_heads"], "--seed", seed, "--tol", t, "--secs", secs, "--reps", reps,
           "--cold-bytes", cold_bytes, "--check-toks", check_toks]  # fmt: skip
    r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=timeout)
    rows = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
    err = next((x for x in rows if x.get("kind") == "error"), None)
    if err or not rows:
        raise HarnessError((err or {}).get("error") or r.stderr.strip()[-2000:] or f"exit {r.returncode}")
    return next(x for x in rows if x["kind"] == "check"), [x for x in rows if x["kind"] == "sample"]


def run1_splits(c, shape, sms=A100_SMS):
    """The split count of the A100 run-1 kernels (two waves of CTAs, >= 256 tokens per split, at most 64): the
    ablation baseline for the current rule (splits_for)."""
    sh = {**PROBLEM, **{k: c[k] for k in PROBLEM if k in c}, **(shape or {})}
    kvh = 1 if c["mla"] else sh["kv_heads"]
    ctas = _cdiv(sh["nq"] * (sh["heads"] // kvh), 16 * c["wm"]) * kvh
    ns = _cdiv(2 * sms, ctas) if ctas < 2 * sms else 1
    return max(1, min(ns, max(1, sh["nkv"] // 256), 64))


def summarize(samples):
    us = [s["us"] for s in samples]
    m = sum(us) / len(us)
    sd = (sum((u - m) ** 2 for u in us) / max(1, len(us) - 1)) ** 0.5
    best = min(samples, key=lambda s: s["us"])
    uj = [s["uJ"] for s in samples if s["uJ"] == s["uJ"]]
    return {"us": m, "us_sd": sd, "GBps": best["GBps"], "TFLOPs": best["TFLOPs"], "uJ": sum(uj) / len(uj) if uj else float("nan"),
            "layers": best["layers"]}  # fmt: skip


def tune(spec, space, harness, objective="speed", secs=0.3, reps=3, cold_bytes=3e8, log=print):
    """Sweep `space` over the spec's problem on the local GPU (correctness-gated). Returns results sorted by objective."""
    key = {"speed": "us", "energy": "uJ"}[objective]
    names, out, seen = list(space), [], set()
    for combo in itertools.product(*(space[k] for k in names)):
        ov = dict(zip(names, combo))
        try:
            c = resolve(spec, ov)
        except SpecError as e:
            log(f"skip {ov}: {e}")
            continue
        if config_key(c) in seen:
            continue
        seen.add(config_key(c))
        try:
            chk, samples = gpu_run(harness, nvcc_build(c)[0], c, secs=secs, reps=reps, cold_bytes=cold_bytes)
        except (GpuBuildError, HarnessError, subprocess.TimeoutExpired) as e:
            log(f"FAIL {label(c)}: {str(e).splitlines()[0]}")
            continue
        if chk["status"] != "ok":
            log(f"FAIL {label(c)}: relerr {chk['relerr']:.2e}")
            continue
        s = summarize(samples)
        out.append({"config": c, **s, "splits": chk["splits"]})
        log(f"{label(c)}  {s['us']:9.2f} us ±{s['us_sd']:.2f}  {s['GBps']:7.1f} GB/s  {s['TFLOPs']:6.2f} TF/s  {s['uJ']:8.1f} uJ")
    out.sort(key=lambda r: (r[key] != r[key], r[key]))
    return out


# Decode / prefill matrix for the A100 session: model shapes x context x KV format.
MATRIX_MODELS = {
    "llama3-8b": {"heads": 32, "kv_heads": 8, "dk": 128},
    "qwen3-1.7b": {"heads": 16, "kv_heads": 8, "dk": 128},
    "mla-dsv2-lite": {"heads": 16, "kv_heads": 1, "dk": 576},
}
MATRIX_CONTEXTS = (1024, 4096, 16384, 32768)


def ablations(c, shape):
    """Phase-2 changes one at a time, each set back to its A100 run-1 value, plus the opt-in fused merge, plus all
    run-1 settings at once: {label: overrides} for config `c` on problem `shape`."""
    run1 = {"split": run1_splits(c, shape)}
    out = {"run-1 splits": dict(run1)}
    if c["kv"] == "q8_0":
        out["deq 0 (smem pass, run 1)"] = {"deq": 0}
        run1["deq"] = 0
    if c["kv"] == "bf16" and c["qsplit"]:
        out["qsplit 0 (run 1)"] = {"qsplit": 0}
        run1["qsplit"] = 0
    if c["mla"] and (c["tk"], c["wn"]) != (32, 2):
        out["tk 32 wn 2 (run 1)"] = {"tk": 32, "wn": 2}
        run1.update(tk=32, wn=2)
    out["merge 1 (fused)"] = {"merge": 1}
    out["run-1 config"] = run1
    return out


ABLATE_MODELS = ("llama3-8b", "mla-dsv2-lite")
ABLATE_KVS = ("f16", "q8_0", "bf16")


def _cell(harness, c, ctx_row, secs, reps, **kw):
    lib = nvcc_build(c)[0]
    chk, samples = gpu_run(harness, lib, c, secs=secs, reps=reps, **kw)
    return {**ctx_row, "impl": "kurn", "config": label(c), "relerr": chk["relerr"], "status": chk["status"], "splits": chk["splits"],
            "device": chk["device"], "sm": chk["sm"], "native": f"sm_{chk['sm']}" in fatbin_contents(lib)[0],
            **summarize(samples)}  # fmt: skip


def matrix(harness, results, kvs=None, models=MATRIX_MODELS, contexts=MATRIX_CONTEXTS, nqs=(1,), secs=0.4, reps=5,
           kernels=None, log=print, arch=ARCH, ggml=None, ablate=(), ablate_models=ABLATE_MODELS):  # fmt: skip
    """For each (model, KV format, context, nq) on tier tier_for(arch): the default kernel (or the
    `kernels[(model, kv)]` overrides) is checked on the GPU and timed in the cold regime -> RESULTS/attn_matrix.jsonl.
    `ggml`: a bench_ggml_attn executable, run on every cell (same data, same timing) -> attn_baselines.jsonl.
    `ablate`: contexts at which every ablations() variant of `ablate_models` runs too -> attn_ablation.jsonl.
    Returns the default rows."""
    os.makedirs(results, exist_ok=True)
    kvs = kvs or SCHEDULE["kv"]({"arch": tier_for(arch)})  # fp8 on the sm_100 / sm_120 tiers
    rows = []
    errs = (SpecError, GpuBuildError, HarnessError, subprocess.TimeoutExpired)
    with open(os.path.join(results, "attn_matrix.jsonl"), "w") as fh, open(os.path.join(results, "attn_baselines.jsonl"), "a") as bh, \
            open(os.path.join(results, "attn_ablation.jsonl"), "w") as ah:  # fmt: skip
        for (mname, m), kv, ctx, nq in itertools.product(models.items(), kvs, contexts, nqs):
            ov = {"kv": kv, "dk": m["dk"], "heads": m["heads"], "kv_heads": m["kv_heads"], "nkv": ctx, "nq": nq, "arch": tier_for(arch)}
            ov.update((kernels or {}).get((mname, kv), {}))
            base = {"model": mname, "kv": kv, "nkv": ctx, "nq": nq}
            c = None
            try:
                c = resolve({}, ov)
                row = {**_cell(harness, c, base, secs, reps), "variant": "default", "splits_predicted": splits_for(c)}
            except errs as e:
                row = {**base, "impl": "kurn", "variant": "default", "status": "error", "error": str(e).splitlines()[0]}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            log(json.dumps(row))
            if ggml and c is not None and kv != "fp8":
                try:
                    chk, samples = ggml_run(ggml, c, secs=secs, reps=reps)
                    b = {
                        **base,
                        "impl": "ggml-cuda",
                        "relerr": chk["relerr"],
                        "status": chk["status"],
                        "device": chk["device"],
                        **summarize(samples),
                    }
                except errs as e:
                    b = {**base, "impl": "ggml-cuda", "status": "error", "error": str(e).splitlines()[0]}
                bh.write(json.dumps(b) + "\n")
                bh.flush()
                log(json.dumps(b))
            if c is not None and ctx in ablate and mname in ablate_models and kv in ABLATE_KVS and nq == 1:
                for vname, vov in ablations(c, None).items():
                    try:
                        vc = resolve({}, {**ov, **vov})
                        a = {**_cell(harness, vc, base, secs, reps), "variant": vname}
                    except errs as e:
                        a = {**base, "impl": "kurn", "variant": vname, "status": "error", "error": str(e).splitlines()[0]}
                    ah.write(json.dumps(a) + "\n")
                    ah.flush()
                    log(json.dumps(a))
    return rows
