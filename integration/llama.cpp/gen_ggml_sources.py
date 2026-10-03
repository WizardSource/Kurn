#!/usr/bin/env python3
"""Generate the kurn kernels for the ggml-cpu KURN extra buffer type.

Formats are discovered from the kurn kernel registry: every weight format that has
both a `gemv` and a `verify` kernel for avx512_vnni with the interleaved `i16`
layout, whose activation format is ggml's Q8_0 or Q8_K, and whose name is a ggml
type (`GGML_TYPE_<NAME>` in ggml.h) is included. A workstream that registers a new
recipe (kurn.ext) therefore gets a llama.cpp kernel without touching this file.

For every format this writes into DIR (normally ggml/src/ggml-cpu/kurn/):
  kurn_<fmt>_gemv.c        decode GEMV (one activation column) + packing glue
  kurn_<fmt>_vfy{2,4,8}.c  multi-column kernels (2-8 activation columns), same
                           packed layout and the same per-column arithmetic
plus kurn.h, kurn_decls.h (prototypes) and kurn_dispatch.h (table: ggml type -> entry points).

    python gen_ggml_sources.py DIR [--ggml-h ggml/include/ggml.h] [--config tuned.json] [--only q8_0,q4_0]

--config is JSON {format: {codegen keys}} (e.g. picked with `kurn tune` or the e2e
harness); the verify kernels inherit the GEMV's algorithm keys so the arithmetic of
a column never depends on how many columns are computed together.
"""

import argparse
import json
import math
import os
import re
import sys

from kurn import spec
from kurn.formats import FORMATS
from kurn.kernels import KERNELS, generate, kernel
from kurn.toolchain import data_path

TARGET = "avx512_vnni"
LAYOUT = "i16"
VFY_COLS = (2, 4, 8)
ACT_TYPES = {"q8_0": ("GGML_TYPE_Q8_0", 32), "q8_K": ("GGML_TYPE_Q8_K", 256)}
DEFAULT_KEYS = {"layout": LAYOUT, "rows": 2, "prefetch": 0}
# Per-format defaults measured through the buffer type (benchmarks/v0.2/q4fix, Qwen3-1.7B shapes,
# 8 threads): 4 row groups per pass = 4 weight streams per core, which is what the DRAM
# prefetchers need (1 stream: 42 GB/s, 4: 80 GB/s on the dev VM); the pair / perm unpacks halve
# the ALU work per 64-byte load. --config overrides these.
TUNED_KEYS = {
    "q8_0": {"rows": 8},  # v0.1 vnni16's tuned pass width, on the i16 records
    "q4_0": {"unpack": "pair", "rows": 4},
    "q4_K": {"unpack": "pair", "correction": "dpmin", "rows": 4},
    "iq4_nl": {"unpack": "perm", "rows": 4},
}
GUARD = "#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)"
# The generated kernels keep per-call activation buffers on the stack: K/32 <= 1024.
MAX_K = 32768


def ggml_type_name(fmt):
    return "GGML_TYPE_" + fmt.upper()


def ggml_types(ggml_h):
    if not ggml_h:
        return None
    return set(re.findall(r"\b(GGML_TYPE_\w+)\s*=", open(ggml_h).read()))


def discover(types=None, only=None):
    """Formats with gemv + verify kernels for the i16 layout (see module doc)."""
    out = []
    for (op, w), k in KERNELS.items():
        if op != "verify" or TARGET not in k.targets:
            continue
        g = KERNELS.get(("gemv", w))
        if g is None or TARGET not in g.targets or FORMATS[w].act not in ACT_TYPES:
            continue
        if types is not None and ggml_type_name(w) not in types:
            continue
        if only and w not in only:
            continue
        try:
            spec.resolve({"op": "gemv", "weights": w, "target": TARGET, "layout": LAYOUT})
        except spec.SpecError:
            continue
        out.append(w)
    return out


def _short(entry):
    return entry[1:].rsplit("_", 1)[0]  # kq8_gemv -> q8


def _glue(src, entry):
    """Packing glue appended to a generated GEMV: size of the packed layout, packing into
    caller-owned memory (the tensor's buffer), and a header that points at it."""
    rec = re.search(r"#define REC_BYTES (\d+)", src)
    dims = re.search(r"pk->nrec_k = ([^;]+); pk->ngroups = ([^;]+);", src)
    if not rec or "typedef struct { int64_t nrec_k, ngroups; uint8_t *buf; } packed_t;" not in src or not dims:
        raise ValueError(f"{entry}: generated GEMV does not have the expected i16 packed_t layout")
    nrec, ngr = dims.groups()
    return f"""
/* --- packing glue (gen_ggml_sources.py) --- */
size_t {entry}_kurn_bytes(int64_t K, int64_t N) {{
    const int64_t nrec_k = {nrec}, ngroups = {ngr};
    return ((size_t)REC_BYTES * nrec_k * ngroups + 63) & ~(size_t)63;
}}
void *{entry}_kurn_view(void *buf, int64_t K, int64_t N) {{
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = {nrec}; pk->ngroups = {ngr}; pk->buf = (uint8_t *)buf;
    return pk;
}}
void {entry}_kurn_pack(const void *W, int64_t K, int64_t N, void *dst) {{
    packed_t *pk = (packed_t *){entry}_prepare(W, K, N);
    memcpy(dst, pk->buf, (size_t)REC_BYTES * pk->nrec_k * pk->ngroups);
    free(pk->buf);
    free(pk);
}}
"""


def build(fmt, keys):
    c = spec.resolve({"op": "gemv", "weights": fmt, "target": TARGET, **keys})
    if c["layout"] != LAYOUT:
        raise ValueError(f"{fmt}: the buffer type needs layout={LAYOUT}, got {c['layout']}")
    entry = kernel(c).entry
    gemv = generate(dict(c, xprep=1))  # + shared activation prep entry points where the lowering has them
    gemv += _glue(gemv, entry)
    # the same GEMV with one row group per pass (same packed layout; only the pass width differs) for
    # range remainders, so rows can be split at record granularity instead of record x rows
    g1 = generate(dict(spec.resolve({"op": "gemv", "weights": fmt, "target": TARGET, **dict(keys, rows=1)}), xprep=1))
    g1 = re.sub(rf"\b{entry}(_prepare|_packed|_packed_x|_xprep|_xprep_bytes)\b", rf"{entry}1\1", g1)
    vkeys = {k: c[k] for k in ("unpack", "correction", "scales", "accum", "ilv") if k in c}
    vbase = kernel({"op": "verify", "weights": fmt}).entry
    vfy = {}
    for cols in VFY_COLS:
        vc = spec.resolve(
            {
                "op": "verify",
                "weights": fmt,
                "target": TARGET,
                "layout": LAYOUT,
                "rows": 1,
                "cols": cols,
                "prefetch": c["prefetch"],
                **vkeys,
            }
        )
        src = generate(dict(vc, xprep=1))
        new = f"{vbase}{cols}"
        src = re.sub(rf"\b{vbase}(_prepare|_packed|_packed_x|_xprep|_xprep_bytes)?\b", rf"{new}\1", src)
        vfy[cols] = (new, src)
    return c, entry, gemv, vfy, g1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--ggml-h", help="ggml/include/ggml.h of the target checkout (formats ggml lacks are skipped)")
    ap.add_argument("--config", help="JSON file or inline JSON {format: {codegen keys}}")
    ap.add_argument("--only", help="comma-separated formats")
    a = ap.parse_args(argv)
    cfg = (json.loads(a.config) if a.config.lstrip().startswith("{") else json.load(open(a.config))) if a.config else {}
    only = set(a.only.split(",")) if a.only else None
    fmts = discover(ggml_types(a.ggml_h), only)
    if not fmts:
        sys.exit("no kurn formats to generate")
    os.makedirs(a.dir, exist_ok=True)
    for f in os.listdir(a.dir):
        if f.startswith("kurn_") and f.endswith(".c"):
            os.remove(os.path.join(a.dir, f))
    with open(data_path("kurn.h")) as fh:
        open(os.path.join(a.dir, "kurn.h"), "w").write(fh.read())

    def write(name, src):
        open(os.path.join(a.dir, name), "w").write(f'{GUARD}\n#include "kurn_decls.h"\n{src}\n#endif\n')

    rows, decls = [], []
    for fmt in fmts:
        keys = {**DEFAULT_KEYS, **TUNED_KEYS.get(fmt, {}), **cfg.get(fmt, {})}
        c, entry, gemv, vfy, g1 = build(fmt, keys)
        write(f"kurn_{fmt}_gemv.c", gemv)
        write(f"kurn_{fmt}_gemv1.c", g1)
        for cols, (_, src) in vfy.items():
            write(f"kurn_{fmt}_vfy{cols}.c", src)
        act_type, act_block = ACT_TYPES[FORMATS[fmt].act]
        period = math.lcm(FORMATS[fmt].block, act_block, 32)
        decls.append(
            f"size_t {entry}_kurn_bytes(int64_t, int64_t); void *{entry}_kurn_view(void *, int64_t, int64_t); "
            f"void {entry}_kurn_pack(const void *, int64_t, int64_t, void *);"
        )
        decls.append(
            f"void *{entry}_prepare(const void *, int64_t, int64_t); "
            f"void {entry}_packed(const void *, const void *, float *, int64_t, int64_t, int64_t); "
            f"void {entry}1_packed(const void *, const void *, float *, int64_t, int64_t, int64_t); "
            f"void *{entry}1_prepare(const void *, int64_t, int64_t);"
        )
        vfy_names = [n for n, _ in vfy.values()]
        xp = f"{entry}_xprep_bytes" in gemv and all(f"{n}_packed_x" in src for n, src in vfy.values())
        if xp:
            decls.append(
                f"size_t {entry}_xprep_bytes(int64_t, int64_t); "
                f"void {entry}_xprep(const void *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void *); "
                f"void {entry}_packed_x(const void *, const void *, const void *, int64_t, int64_t, float *, int64_t, int64_t, int64_t); "
                f"void {entry}1_packed_x(const void *, const void *, const void *, int64_t, int64_t, float *, int64_t, int64_t, int64_t);"
            )
            for n in [f"{entry}1"] + vfy_names:
                decls.append(f"size_t {n}_xprep_bytes(int64_t, int64_t); "
                             f"void {n}_xprep(const void *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void *);")
        for name, _ in vfy.values():
            decls.append(f"void {name}(const void *, const void *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);")
            decls.append(
                f"void *{name}_prepare(const void *, int64_t, int64_t); "
                f"void {name}_packed(const void *, const void *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);"
            )
            if xp:
                decls.append(f"void {name}_packed_x(const void *, const void *, const void *, int64_t, int64_t, float *, int64_t, "
                             "int64_t, int64_t, int64_t, int64_t);")
        desc = " ".join(f"{k}={c[k]}" for k in ("layout", "rows", "prefetch", "unpack", "correction", "scales", "accum", "ilv"))
        rec_rows = 32 if c["unpack"] == "pair" else 16  # pair records hold rows r and r + 16 in one byte
        align = rec_rows * int(c.get("ilv", 1))
        rows.append(
            f'    {{ {ggml_type_name(fmt)}, {act_type}, {period}, {align}, {rec_rows * int(c["rows"])}, "{fmt}", "{desc}", '
            f"{entry}_kurn_bytes, "
            f"{entry}_kurn_view, {entry}_kurn_pack, {entry}_packed, {entry}1_packed, "
            f"{{ {', '.join(n + '_packed' for n, _ in vfy.values())} }}, "
            + (f"{entry}_xprep_bytes, {entry}_xprep, {entry}_packed_x, {entry}1_packed_x, "
               f"{{ {', '.join(n + '_packed_x' for n in vfy_names)} }} }},"
               if xp else "NULL, NULL, NULL, NULL, { NULL, NULL, NULL } },")
        )
        print(f"{fmt}: {desc}")
    cpp_open = ["#ifdef __cplusplus", 'extern "C" {', "#endif"]
    cpp_close = ["#ifdef __cplusplus", "}", "#endif"]
    gen = "// Generated by kurn/integration/llama.cpp/gen_ggml_sources.py. Do not edit."
    decl_h = [gen, "#pragma once", "#include <stddef.h>", "#include <stdint.h>", *cpp_open, *decls, *cpp_close]
    open(os.path.join(a.dir, "kurn_decls.h"), "w").write("\n".join(decl_h) + "\n")
    hdr = [
        gen,
        "#pragma once",
        '#include "ggml.h"',
        '#include "kurn_decls.h"',
        *cpp_open,
        f"#define KURN_MAX_K {MAX_K}",
        "#define KURN_VFY_MAX 8",
        "typedef void (*kurn_gemv_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t);",
        "typedef void (*kurn_vfy_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);",
        "// shared activation prep: xprep(X, K, C, m0, m1, k0, k1, ws) fills the per-block activation tables of",
        "// columns [m0, m1) and 32-value blocks [k0, k1) into ws (xprep_bytes(K, C) bytes); the _x kernels read them",
        "typedef void (*kurn_gemv_x_fn)(const void *, const void *, const void *, int64_t, int64_t, float *, int64_t, int64_t, int64_t);",
        "typedef void (*kurn_vfy_x_fn)(const void *, const void *, const void *, int64_t, int64_t, float *, int64_t, int64_t, int64_t,",
        "                              int64_t, int64_t);",
        "typedef struct {",
        "    int type, vec_dot_type;  // enum ggml_type",
        "    int64_t k_multiple;      // K must be a multiple of this",
        "    int64_t row_align;       // row granularity of the kernels (rows of one record, x ilv)",
        "    int64_t row_pass;        // rows of one pass of `gemv` (record rows x row groups); `gemv1` does one record",
        "    const char *name, *config;",
        "    size_t (*bytes)(int64_t K, int64_t N);",
        "    void *(*view)(void *buf, int64_t K, int64_t N);",
        "    void (*pack)(const void *W, int64_t K, int64_t N, void *dst);",
        "    kurn_gemv_fn gemv, gemv1;",
        f"    kurn_vfy_fn vfy[{len(VFY_COLS)}];  // {', '.join(map(str, VFY_COLS))} columns",
        "    size_t (*xprep_bytes)(int64_t K, int64_t C);  // NULL: the format's kernels prep per call",
        "    void (*xprep)(const void *X, int64_t K, int64_t C, int64_t m0, int64_t m1, int64_t k0, int64_t k1, void *ws);",
        "    kurn_gemv_x_fn gemv_x, gemv1_x;",
        f"    kurn_vfy_x_fn vfy_x[{len(VFY_COLS)}];",
        "} kurn_kernel;",
        f"static const int kurn_vfy_cols[{len(VFY_COLS)}] = {{ {', '.join(map(str, VFY_COLS))} }};",
        "static const kurn_kernel kurn_kernels[] = {",
        *rows,
        "};",
        *cpp_close,
    ]
    open(os.path.join(a.dir, "kurn_dispatch.h"), "w").write("\n".join(hdr) + "\n")


if __name__ == "__main__":
    main()
