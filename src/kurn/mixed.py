"""Per-tensor / per-row mixed precision guided by an outlier/sensitivity predictor (`kurn mix`).

The predictor follows the Project's KV-outlier work: the best single feature there was a
*format-aware isolated simulation* (quantize one layer in the target format, measure its
relative output error). For weights, the same feature is cheap and needs no forward pass:

    err_t(f) = sum_j imp_j * ||W_t[:, j] - Q_f(W_t)[:, j]||^2 / sum_j imp_j * ||W_t[:, j]||^2

where imp_j = E[x_j^2] is llama.cpp's importance matrix (diagonal of the input Hessian) and
Q_f is ggml's own quantizer for format f (read back from `--pure` quantized GGUFs, so the
error is exactly what llama.cpp would produce). It estimates the relative output error of
tensor t in isolation. Tier-1 features (weights/activations only: per-column kurtosis,
channel-importance ratio max/median, max|w|/rms) are recorded for comparison.

Allocation: minimise sum_t g_t * err_t(f_t) subject to sum_t bits_t(f_t) <= budget, solved by
the greedy Lagrangian on each tensor's lower convex hull (optimal on the hull points), where
g_t is 1 ("rel") or a per-tensor-kind gain. The recipe is applied with llama-quantize's
`--tensor-type-file` (anchored regexes), so mixes run on stock llama.cpp.

GGUF helpers (`tensors`, `dequant`, `write_gguf`) are shared with codebook.py / lowrank.py.
"""

import heapq
import json
import math
import os
import re
import subprocess

import numpy as np

# ---------------------------------------------------------------- GGUF I/O


def _gguf():
    import gguf  # optional dependency (the `gguf` package from llama.cpp)

    return gguf


def reader(path):
    return _gguf().GGUFReader(path)


def tensors(path_or_reader):
    r = reader(path_or_reader) if isinstance(path_or_reader, str) else path_or_reader
    return {t.name: t for t in r.tensors}


def type_name(t):
    return t.tensor_type.name


def bits_per_weight(tname):
    gguf = _gguf()
    q = gguf.GGMLQuantizationType[tname.upper()]
    blk, nbytes = gguf.GGML_QUANT_SIZES[q]
    return 8.0 * nbytes / blk


def dequant(t, rows=None):
    """ReaderTensor -> float32 (n_rows, n_cols); `rows` = slice of rows (row-chunked reads)."""
    gguf = _gguf()
    data = t.data if rows is None else t.data[rows]
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.asarray(data, dtype=np.float32).reshape(data.shape[0], -1) if data.ndim > 1 else np.asarray(data)
    if t.tensor_type == gguf.GGMLQuantizationType.F16:
        return np.asarray(data, dtype=np.float32)
    return gguf.quants.dequantize(np.asarray(data), t.tensor_type).astype(np.float32, copy=False)


def n_rows(t):
    return int(t.shape[-1]) if len(t.shape) > 1 else 1


def n_cols(t):
    return int(t.shape[0])


def row_chunks(t, chunk=8192):
    n = n_rows(t)
    for a in range(0, n, chunk):
        yield slice(a, min(n, a + chunk))


def is_matrix(name, t):
    return len(t.shape) == 2 and name.endswith(".weight") and "norm" not in name


def load_imatrix(path):
    """llama.cpp imatrix (GGUF format) -> {tensor name: mean squared activation per input column}."""
    out = {}
    ts = tensors(path)
    for name, t in ts.items():
        if name.endswith(".in_sum2"):
            base = name[: -len(".in_sum2")]
            cnt = float(np.asarray(ts[base + ".counts"].data).reshape(-1)[0])
            out[base] = np.asarray(t.data, dtype=np.float64).reshape(-1) / max(cnt, 1.0)
    return out


def write_gguf(src, out, replace, dtype="F16", raw=None, qtypes=None):
    """Copy GGUF `src` to `out`, replacing the tensors named in `replace` ({name: float32 array
    (rows, cols) or a zero-argument callable returning one}) by `dtype` (F16, F32 or a type
    gguf-py can quantize, e.g. Q8_0), and those in `raw` ({name: ReaderTensor of another GGUF})
    by that tensor as stored there. Tensors also named in `qtypes` ({name: type}) are written as
    that type and their `replace` value must be the already-quantized bytes (rows, row_bytes).
    Other tensors are copied as stored. Streams one tensor at a time, so memory stays at about
    one tensor."""
    gguf = _gguf()
    r = reader(src)
    raw = raw or {}
    qtypes = {n: gguf.GGMLQuantizationType[q.upper()] for n, q in (qtypes or {}).items()}
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(out, arch=arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(name, f.contents(), vt, sub_type=sub)
    qt = gguf.GGMLQuantizationType[dtype]
    blk, tsz = gguf.GGML_QUANT_SIZES[qt]
    srcs = [raw.get(t.name, t) for t in r.tensors]
    for t in srcs:
        if t.name in qtypes:
            rows, cols = n_rows(t), n_cols(t)
            b, s = gguf.GGML_QUANT_SIZES[qtypes[t.name]]
            w.add_tensor_info(t.name, (rows, cols // b * s), np.dtype(np.uint8), rows * cols // b * s, raw_dtype=qtypes[t.name])
        elif t.name in replace:
            rows, cols = n_rows(t), n_cols(t)
            nbytes = rows * cols // blk * tsz
            if qt in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                w.add_tensor_info(t.name, (rows, cols), np.dtype(np.float16 if tsz == 2 else np.float32), nbytes)
            else:
                w.add_tensor_info(t.name, (rows, cols * tsz // blk), np.dtype(np.uint8), nbytes, raw_dtype=qt)
        else:
            shape = tuple(int(s) for s in reversed(t.shape))
            if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                w.add_tensor_info(t.name, shape, np.asarray(t.data).dtype, int(t.n_bytes), raw_dtype=t.tensor_type)
            else:  # gguf-py takes the byte shape of quantized / BF16 tensors
                b, s = gguf.GGML_QUANT_SIZES[t.tensor_type]
                w.add_tensor_info(t.name, shape[:-1] + (shape[-1] // b * s,), np.dtype(np.uint8), int(t.n_bytes),
                                  raw_dtype=t.tensor_type)  # fmt: skip
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in srcs:
        if t.name in replace:
            v = replace[t.name]
            if t.name in qtypes:
                w.write_tensor_data(np.ascontiguousarray(v() if callable(v) else v, dtype=np.uint8))
                continue
            a = np.ascontiguousarray(v() if callable(v) else v, dtype=np.float32)
            if qt == gguf.GGMLQuantizationType.F16:
                a = a.astype(np.float16)
            elif qt != gguf.GGMLQuantizationType.F32:
                a = gguf.quants.quantize(a, qt)
            w.write_tensor_data(a)
        else:
            w.write_tensor_data(np.asarray(t.data))
    w.close()
    return out


# ---------------------------------------------------------------- ggml's quantizers in-process


def _libggml():
    import ctypes

    path = os.environ.get("KURN_LIBGGML", os.path.expanduser("~/src/llama.cpp/build/bin/libggml-base.so"))
    lib = ctypes.CDLL(path)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]  # fmt: skip
    lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    return lib


_LIB = []


def ggml_quantize(W, tname, imp=None, threads=4):
    """Quantize float rows W (rows, cols) with ggml's own quantizer for `tname` (what llama-quantize
    does per tensor, imatrix = mean squared activation per column). Returns raw bytes (rows, row_bytes)."""
    from concurrent.futures import ThreadPoolExecutor

    gguf = _gguf()
    if not _LIB:
        _LIB.append(_libggml())
    lib = _LIB[0]
    qt = gguf.GGMLQuantizationType[tname.upper()]
    blk, tsz = gguf.GGML_QUANT_SIZES[qt]
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    rb = cols // blk * tsz
    out = np.empty((rows, rb), dtype=np.uint8)
    if imp is None and lib.ggml_quantize_requires_imatrix(int(qt)):
        imp = np.ones(cols)
    im = None if imp is None else np.ascontiguousarray(imp, dtype=np.float32)
    step = max(1, min(256, rows // max(1, threads)))

    def run(a):
        n = min(step, rows - a)
        lib.ggml_quantize_chunk(int(qt), W[a:].ctypes.data, out[a:].ctypes.data, 0, n, cols, None if im is None else im.ctypes.data)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(run, range(0, rows, step)))
    return out


def ggml_roundtrip(W, tname, imp=None, threads=4):
    """Dequantized ggml quantization of W (float32, same shape)."""
    return _gguf().quants.dequantize(ggml_quantize(W, tname, imp, threads), _gguf().GGMLQuantizationType[tname.upper()])


# ---------------------------------------------------------------- profile (predictor features)


def kind(name):
    """'blk.3.ffn_down.weight' -> 'ffn_down'; 'token_embd.weight' -> 'token_embd'."""
    parts = name.split(".")
    return parts[2] if parts[0] == "blk" else parts[0]


def layer(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def _kurtosis_cols(W):
    m = W.mean(axis=0)
    c = W - m
    v = (c * c).mean(axis=0) + 1e-30
    return ((c**4).mean(axis=0) / (v * v))


def profile(ref, quants, imatrix=None, rows_out=None, log=print, threads=4):
    """Per-tensor predictor table.

    ref: BF16/F16 GGUF path; quants: {type name: GGUF path quantized with --pure, or None to run
    ggml's quantizer in-process (same per-tensor result, no full-model file)};
    imatrix: llama.cpp imatrix path (None: uniform importance).
    rows_out: optional .npz path for per-row errors (per-row / channel mixing).
    Returns {"tensors": {name: {...}}, "types": {type: bpw}}."""
    imx = load_imatrix(imatrix) if imatrix else {}
    rt = tensors(ref)
    qts = {q: (tensors(p) if p else None) for q, p in quants.items()}
    out, row_err = {}, {}
    for name, t in rt.items():
        if not is_matrix(name, t):
            continue
        base = name[: -len(".weight")] + ".weight"
        imp = imx.get(base)
        has_imp = imp is not None
        if imp is None:
            imp = np.ones(n_cols(t))
        imp = imp.astype(np.float64)
        num = {q: 0.0 for q in quants}
        rows = {q: [] for q in quants}
        den, wmax, ssq, kurt_acc, nkurt = 0.0, 0.0, 0.0, 0.0, 0
        types = {}
        for sl in row_chunks(t):
            W = dequant(t, sl).astype(np.float64)
            w2 = W * W
            rden = w2 @ imp
            den += float(rden.sum())
            ssq += float(w2.sum())
            wmax = max(wmax, float(np.abs(W).max()))
            if nkurt < 4096:
                k = _kurtosis_cols(W)
                kurt_acc = max(kurt_acc, float(k.max()))
                nkurt += W.shape[0]
            for q, qt in qts.items():
                if qt is None:
                    types[q] = q.upper()
                    D = ggml_roundtrip(W, q, imx.get(base), threads).astype(np.float64) - W
                else:
                    types[q] = type_name(qt[name])
                    D = dequant(qt[name], sl).astype(np.float64) - W
                re_ = (D * D) @ imp
                num[q] += float(re_.sum())
                if rows_out:
                    rows[q].append(re_ / np.maximum(rden, 1e-30))
        n = int(t.n_elements)
        out[name] = {
            "kind": kind(name), "layer": layer(name), "rows": n_rows(t), "cols": n_cols(t), "params": n,
            "has_imatrix": has_imp,
            "err": {q: num[q] / max(den, 1e-30) for q in quants},
            "abs_err": {q: num[q] for q in quants},
            "stored_type": types,
            "feat": {
                "max_over_rms": wmax / math.sqrt(ssq / n),
                "col_kurtosis_max": kurt_acc,
                "imp_max_over_median": float(imp.max() / max(np.median(imp), 1e-30)),
                "imp_mean": float(imp.mean()),
            },
        }  # fmt: skip
        if rows_out:
            row_err[name] = np.stack([np.concatenate(rows[q]) for q in quants]).astype(np.float32)
        log(f"{name:32s} " + " ".join(f"{q}={out[name]['err'][q]:.2e}" for q in quants))
    if rows_out:
        np.savez_compressed(rows_out, types=np.array(list(quants)), **row_err)
    return {"tensors": out, "types": {q: bits_per_weight(q) for q in quants}}


# ---------------------------------------------------------------- allocation


def _hull(points):
    """Lower convex hull of (bits, cost) points sorted by bits, dropping dominated ones."""
    pts = sorted(points)
    clean = []
    for p in pts:
        if clean and p[1] >= clean[-1][1]:
            continue  # more bits, not better
        clean.append(p)
    hull = []
    for p in clean:
        while len(hull) >= 2:
            (b1, c1, _), (b2, c2, _) = hull[-2], hull[-1]
            if (c2 - c1) * (p[0] - b1) >= (p[1] - c1) * (b2 - b1):  # hull[-1] above the chord
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def gains(prof, how="rel"):
    """Per-tensor weight g_t of the objective."""
    ts = prof["tensors"]
    if how == "rel":
        return {n: 1.0 for n in ts}
    if how == "abs":  # absolute output-error proxy (imatrix-weighted SSE)
        return {n: 1.0 for n in ts}
    if isinstance(how, dict):  # per kind
        return {n: float(how.get(v["kind"], 1.0)) for n, v in ts.items()}
    raise ValueError(how)


def allocate(prof, bpw, types=None, how="rel", fixed=None):
    """Greedy Lagrangian allocation. Returns ({tensor: type}, achieved bpw, objective).

    types: candidate formats (default: all profiled); fixed: {tensor: type} pinned choices."""
    ts = prof["tensors"]
    types = [q for q in (types or prof["types"]) if q in prof["types"]]
    g = gains(prof, how)
    key = "abs_err" if how == "abs" else "err"
    fixed = fixed or {}
    total_params = sum(v["params"] for v in ts.values())
    budget = bpw * total_params
    hulls, pos, bits = {}, {}, 0.0
    for n, v in ts.items():
        cand = [fixed[n]] if n in fixed else types
        pts = [(prof["types"][q] * v["params"], g[n] * v[key][q], q) for q in cand]
        hulls[n] = _hull(pts)
        pos[n] = 0
        bits += hulls[n][0][0]
    heap = []

    def push(n):
        h, i = hulls[n], pos[n]
        if i + 1 < len(h):
            db, dc = h[i + 1][0] - h[i][0], h[i][1] - h[i + 1][1]
            heapq.heappush(heap, (-dc / db, n, i))

    for n in ts:
        push(n)
    while heap:
        _, n, i = heapq.heappop(heap)
        if pos[n] != i:
            continue
        db = hulls[n][i + 1][0] - hulls[n][i][0]
        if bits + db > budget:
            continue  # cannot afford this step; cheaper steps of other tensors may still fit
        bits += db
        pos[n] = i + 1
        push(n)
    assign = {n: hulls[n][pos[n]][2] for n in ts}
    obj = sum(hulls[n][pos[n]][1] for n in ts)
    return assign, bits / total_params, obj


def mix_bpw(prof, assign):
    ts = prof["tensors"]
    return sum(prof["types"][assign[n]] * v["params"] for n, v in ts.items()) / sum(v["params"] for v in ts.values())


def taalas_rule(prof, bpw, low="Q3_K", high="Q6_K"):
    """Taalas-style two-format mix: everything `low`, the most sensitive tensors (by err_t(low)
    per bit gained) promoted to `high` until the budget is met."""
    return allocate(prof, bpw, types=[low, high])


def recipe_lines(assign):
    """{tensor: type} -> llama-quantize --tensor-type-file lines (anchored regexes; first match wins)."""
    out = []
    for n, q in sorted(assign.items(), key=lambda kv: (layer(kv[0]), kv[0])):
        if n == "token_embd.weight":
            continue  # --token-embedding-type
        out.append(f"^{re.escape(n)}$={q.lower()}")
    return out


def write_recipe(path, assign, meta=None):
    with open(path, "w") as fh:
        fh.write("\n".join(recipe_lines(assign)) + "\n")
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump({"assign": assign, **(meta or {})}, fh, indent=1, sort_keys=True)


def quantize(recipe_json, src, out, imatrix=None, llama_bin=None, default="Q4_K", threads=8, lock=None):
    """Run llama-quantize with a recipe written by write_recipe (its .txt next to the .json)."""
    with open(recipe_json) as fh:
        assign = json.load(fh)["assign"]
    txt = os.path.splitext(recipe_json)[0] + ".txt"
    if not os.path.exists(txt):
        write_recipe(txt, assign)
    exe = os.path.join(llama_bin or os.path.expanduser("~/src/llama.cpp/build/bin"), "llama-quantize")
    cmd = [exe, "--tensor-type-file", txt]
    if "token_embd.weight" in assign:
        cmd += ["--token-embedding-type", assign["token_embd.weight"].lower()]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [src, out, default, str(threads)]
    if lock:
        cmd = [lock] + cmd
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"llama-quantize failed:\n{r.stderr[-3000:]}")
    return out


def quantize_inproc(recipe_json, src, out, imatrix=None, threads=4, log=None):
    """Apply a recipe with ggml's quantizers in-process (libggml via ctypes, same bytes as
    llama-quantize for the same type and imatrix). Streams one tensor at a time."""
    with open(recipe_json) as fh:
        assign = json.load(fh)["assign"]
    ts = tensors(src)
    imx = load_imatrix(imatrix) if imatrix else {}

    def job(n, q):
        def f():
            if log:
                log(f"{n:32s} {q}")
            return ggml_quantize(dequant(ts[n]), q, imx.get(n), threads)
        return f

    return write_gguf(src, out, {n: job(n, q) for n, q in assign.items()}, qtypes=assign)


def gguf_bpw(path):
    """Achieved bits per weight over the 2-D weight matrices of a GGUF (norms excluded)."""
    bits = params = 0
    for name, t in tensors(path).items():
        if is_matrix(name, t):
            params += int(t.n_elements)
            bits += 8 * int(t.n_bytes)
    return bits / params


# ---------------------------------------------------------------- per-row (channel) mixing


def allocate_rows(prof, rows_npz, bpw, types=None, group=16):
    """Per-row-group allocation (output channels): each group of `group` rows of a tensor picks
    its own format. Returns {tensor: array of type index per row group}, types, achieved bpw."""
    z = np.load(rows_npz)
    all_types = [str(s) for s in z["types"]]
    types = [q for q in (types or all_types) if q in all_types]
    ti = [all_types.index(q) for q in types]
    ts = prof["tensors"]
    items = []  # (tensor, group index, cost per type)
    for n, v in ts.items():
        # per-row relative output error, rows weighted equally: the mean over rows is the
        # tensor's error when every row has the same output energy
        E = z[n][ti] / v["rows"]  # (types, rows)
        for gi in range((v["rows"] + group - 1) // group):
            items.append((n, gi, E[:, gi * group : (gi + 1) * group].sum(axis=1), min(group, v["rows"] - gi * group) * v["cols"]))
    total = sum(v["params"] for v in ts.values())
    budget = bpw * total
    bpws = [prof["types"][q] for q in types]
    hulls, pos, bits = [], [], 0.0
    for (_, _, cost, p) in items:
        h = _hull([(bpws[k] * p, float(cost[k]), k) for k in range(len(types))])
        hulls.append(h)
        pos.append(0)
        bits += h[0][0]
    heap = []

    def push(i):
        h, j = hulls[i], pos[i]
        if j + 1 < len(h):
            heapq.heappush(heap, (-(h[j][1] - h[j + 1][1]) / (h[j + 1][0] - h[j][0]), i, j))

    for i in range(len(items)):
        push(i)
    while heap:
        _, i, j = heapq.heappop(heap)
        if pos[i] != j:
            continue
        db = hulls[i][j + 1][0] - hulls[i][j][0]
        if bits + db > budget:
            continue
        bits += db
        pos[i] = j + 1
        push(i)
    out = {}
    for i, (n, _, _, _) in enumerate(items):
        out.setdefault(n, []).append(hulls[i][pos[i]][2])
    return {n: np.array(v, dtype=np.int8) for n, v in out.items()}, types, bits / total


def frontier(prof, bpws, types=None, how="rel"):
    """Predicted objective along a bpw sweep: [(target, achieved bpw, objective, {type: params share})]."""
    total = sum(v["params"] for v in prof["tensors"].values())
    out = []
    for b in bpws:
        assign, got, obj = allocate(prof, b, types, how)
        share = {}
        for n, q in assign.items():
            share[q] = share.get(q, 0.0) + prof["tensors"][n]["params"] / total
        out.append((b, got, obj, share))
    return out


# ---------------------------------------------------------------- `kurn mix` CLI


def cli_main(argv):
    import argparse

    ap = argparse.ArgumentParser(prog="kurn mix", description="per-tensor mixed precision from a sensitivity profile")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="per-tensor error table from --pure quantized GGUFs")
    p.add_argument("ref", help="BF16/F16 GGUF")
    p.add_argument("quants", nargs="+", metavar="TYPE[=GGUF]", help="TYPE alone: quantize in-process with libggml")
    p.add_argument("--imatrix")
    p.add_argument("--rows", help="also save per-row errors (.npz) for per-row-group mixing")
    p.add_argument("-o", "--out", required=True, help="profile .json")
    p = sub.add_parser("plan", help="allocate formats under a bpw budget; writes RECIPE.txt + RECIPE.json")
    p.add_argument("profile")
    p.add_argument("--bpw", type=float, required=True, help="budget over the 2-D weights (incl. token_embd)")
    p.add_argument("--types", help="comma-separated candidates (default: all profiled), e.g. Q3_K,Q6_K")
    p.add_argument("--how", default="rel", choices=["rel", "abs"])
    p.add_argument("--fix", action="append", default=[], metavar="TENSOR=TYPE", help="pin a tensor's format")
    p.add_argument("--weights", metavar="KIND=W,...", help="per-kind weights of the relative errors (overrides --how)")
    p.add_argument("-o", "--out", required=True, help="recipe .txt (llama-quantize --tensor-type-file)")
    p = sub.add_parser("frontier", help="predicted objective vs bpw")
    p.add_argument("profile")
    p.add_argument("--bpw", default="2.5,3,3.5,4,4.5,5,6")
    p.add_argument("--types")
    p.add_argument("--how", default="rel", choices=["rel", "abs"])
    p.add_argument("--weights", metavar="KIND=W,...")
    p = sub.add_parser("quantize", help="apply a recipe with llama-quantize")
    p.add_argument("recipe", help="RECIPE.json written by `plan`")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--imatrix")
    p.add_argument("--llama-bin")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--lock", help="wrapper to run llama-quantize under (benchlock.sh)")
    p.add_argument("--inproc", action="store_true", help="quantize with libggml in-process (streams; no llama-quantize)")
    a = ap.parse_args(argv)
    types = a.types.split(",") if getattr(a, "types", None) else None
    if getattr(a, "weights", None):
        a.how = {k: float(v) for k, v in (s.split("=", 1) for s in a.weights.split(","))}
    if a.cmd == "profile":
        quants = dict((s.split("=", 1) + [None])[:2] for s in a.quants)
        prof = profile(a.ref, quants, a.imatrix, a.rows)
        with open(a.out, "w") as fh:
            json.dump(prof, fh, indent=1)
    elif a.cmd == "plan":
        with open(a.profile) as fh:
            prof = json.load(fh)
        fixed = dict(s.split("=", 1) for s in a.fix)
        assign, got, obj = allocate(prof, a.bpw, types, a.how, fixed)
        write_recipe(a.out, assign, {"bpw_target": a.bpw, "bpw": got, "objective": obj, "types": types, "how": a.how})
        print(f"{a.out}: {got:.4f} bpw, predicted objective {obj:.4g}")
    elif a.cmd == "frontier":
        with open(a.profile) as fh:
            prof = json.load(fh)
        for b, got, obj, share in frontier(prof, [float(s) for s in a.bpw.split(",")], types, a.how):
            mix = " ".join(f"{q}:{s:.0%}" for q, s in sorted(share.items(), key=lambda kv: -kv[1]))
            print(f"{b:5.2f}  {got:.4f} bpw  obj {obj:.4g}  {mix}")
    elif a.cmd == "quantize":
        if a.inproc:
            quantize_inproc(a.recipe, a.src, a.out, a.imatrix, a.threads)
        else:
            quantize(a.recipe, a.src, a.out, a.imatrix, a.llama_bin, threads=a.threads, lock=a.lock)
        print(f"{a.out}: {gguf_bpw(a.out):.4f} bpw (2-D weights)")
    return 0


def compose_rows(ref, quants, row_assign, types, group=16, out=None, dtype="F16"):
    """Write a GGUF whose tensors take each row group from the chosen --pure quantized GGUF
    (dequantized, stored as `dtype`): the quality of a per-row mix in stock llama.cpp."""
    qts = {q: tensors(quants[q]) for q in types}

    def build(n, ga):
        t0 = qts[types[0]][n]
        R = np.empty((n_rows(t0), n_cols(t0)), dtype=np.float32)
        for k, q in enumerate(types):
            sel = np.repeat(ga == k, group)[: R.shape[0]]
            if sel.any():
                R[sel] = dequant(qts[q][n])[sel]
        return R

    rep = {n: (lambda n=n, ga=ga: build(n, ga)) for n, ga in row_assign.items()}
    return write_gguf(ref, out, rep, dtype=dtype)
