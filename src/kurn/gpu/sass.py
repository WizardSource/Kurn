"""SASS inspection without a GPU: instruction mix of the main kernel and of its hottest loop.

`cuobjdump -sass` on an nvcc build; per kernel it counts instruction classes (tensor-core MMA, ldmatrix, cp.async,
global/shared loads, FFMA, f16x2 math, integer logic, barriers) and flags local memory (LDL/STL: spills or stack
arrays). The hot loop is the backward branch whose body holds the most tensor-core / dp4a instructions.
"""

import re
import subprocess

from .toolchain import GpuBuildError, nvcc, nvcc_build

CLASSES = {
    "mma": re.compile(r"^(HMMA|IMMA)"),
    "ldsm": re.compile(r"^LDSM"),
    "cp_async": re.compile(r"^LDGSTS"),
    "ldg": re.compile(r"^LDG"),
    "lds": re.compile(r"^LDS(?!M)"),
    "sts": re.compile(r"^STS"),
    "stg": re.compile(r"^(STG|RED|ATOM)"),
    "local": re.compile(r"^(LDL|STL)"),
    "ffma": re.compile(r"^(FFMA|FMUL|FADD)"),
    "f16x2": re.compile(r"^(HFMA2|HADD2|HMUL2)"),
    "dp4a": re.compile(r"^IDP"),
    "int": re.compile(r"^(LOP3|PRMT|SHF|IADD3|IMAD|LEA|SEL|ISETP|POPC|BMSK|SGXT)"),
    "cvt": re.compile(r"^(F2F|F2I|I2F|F2FP)"),
    "bar": re.compile(r"^(BAR|DEPBAR)"),
    "shfl": re.compile(r"^SHFL"),
}
_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")
_LABEL = re.compile(r"^\s*\.(L_x_\d+):")
_FUNC = re.compile(r"Function : (\S+)")


def ldg_width(op):
    """Bytes per thread of a global load opcode (LDG.E.128 -> 16, LDG.E.U16 -> 2, LDG.E -> 4), else 0."""
    if not op.startswith("LDG"):
        return 0
    for suf, w in ((".128", 16), (".64", 8), (".U16", 2), (".S16", 2), (".U8", 1), (".S8", 1)):
        if suf in op:
            return w
    return 4


def _classify(op):
    base = op.split(".")[0]
    for name, rx in CLASSES.items():
        if rx.match(base):
            return name
    return "other"


def parse(text):
    """cuobjdump -sass text -> {short kernel name: {"counts": {...}, "loop": {...} | None, "total": n}}."""
    out, name, insts, labels = {}, None, [], {}

    def flush():
        if name is None:
            return
        counts = {}
        for _, op, _ in insts:
            counts[_classify(op)] = counts.get(_classify(op), 0) + 1
        loops = []
        for addr, op, args in insts:
            if op.startswith("BRA"):
                m = re.search(r"0x([0-9a-f]+)", args)
                if m and int(m.group(1), 16) < addr:
                    lo = int(m.group(1), 16)
                    body = [x for x in insts if lo <= x[0] <= addr]
                    c = {}
                    for _, o, _ in body:
                        c[_classify(o)] = c.get(_classify(o), 0) + 1
                        if ldg_width(o):
                            c[f"ldg{8 * ldg_width(o)}"] = c.get(f"ldg{8 * ldg_width(o)}", 0) + 1
                    loops.append({"len": len(body), "counts": c})
        hot = max(loops, key=lambda lp: (lp["counts"].get("mma", 0) + lp["counts"].get("dp4a", 0), -lp["len"]), default=None)
        short = next((k for k in ("kg_gemm", "kg_gemv", "kg_repack", "kg_x16", "kg_quant_q8_0", "kg_quant_q8_K") if k in name), name)
        out[short] = {"counts": counts, "total": len(insts), "loop": hot}

    for line in text.splitlines():
        m = _FUNC.search(line)
        if m:
            flush()
            name, insts, labels = m.group(1), [], {}
            continue
        m = _LABEL.match(line)
        if m:
            labels[m.group(1)] = len(insts)
            continue
        m = _LINE.search(line)
        if m and name:
            insts.append((int(m.group(1), 16), m.group(2), m.group(3)))
    flush()
    return out


def inspect(c, arch=None):
    """Build `c` for one arch and return parse() of its SASS."""
    if not nvcc():
        raise GpuBuildError("nvcc not found")
    arch = arch or c["arch"]
    so, _ = nvcc_build(dict(c, arch=arch), (arch,))
    from .cudaenv import cuobjdump

    tool = cuobjdump()
    r = subprocess.run([tool, "-sass", so], capture_output=True, text=True)
    if r.returncode:
        raise GpuBuildError(f"cuobjdump failed: {r.stderr[-500:]}")
    return parse(r.stdout)


def summary(c, rep):
    """One line for the main kernel: whole-kernel counts, local memory, and the hot loop's mix."""
    main = rep.get("kg_gemm" if c["op"] == "gemm" else "kg_gemv")
    if not main:
        return "main kernel not found"
    k, lp = main["counts"], main["loop"]
    s = f"mma={k.get('mma', 0)} ldsm={k.get('ldsm', 0)} cp.async={k.get('cp_async', 0)} local={k.get('local', 0)} insts={main['total']}"
    if lp:
        lc = lp["counts"]
        s += (f" | hot loop: {lp['len']} insts, mma={lc.get('mma', 0)} ldsm={lc.get('ldsm', 0)} lds={lc.get('lds', 0)} "
              f"cp.async={lc.get('cp_async', 0)} ffma={lc.get('ffma', 0)} f16x2={lc.get('f16x2', 0)} int={lc.get('int', 0)} "
              f"dp4a={lc.get('dp4a', 0)} ldg={lc.get('ldg', 0)} (128-bit {lc.get('ldg128', 0)}, 64-bit {lc.get('ldg64', 0)}, "
              f"32-bit {lc.get('ldg32', 0)}, 16-bit {lc.get('ldg16', 0)})")  # fmt: skip
    return s
