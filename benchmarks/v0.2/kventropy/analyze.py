"""analyze.py DUMP.seq OUT.json : entropy models + zstd for K and V, per prefix length, averaged over layers."""
import json, sys, time
import numpy as np
import zstandard
from kvseq import Seq
from kvent import *

path, out = sys.argv[1], sys.argv[2]
s = Seq(path)
fmt = path.rsplit("-", 1)[1].split(".")[0]
prefixes = [p for p in (4096, 8192, 16384, 32768) if p <= s.n]
res = {"n": s.n, "nlayer": s.nlayer, "ktype": s.ktype, "vtype": s.vtype, "fmt": fmt, "prefix": {}}


def zstd_bits(buf, n_el, level=3):
    c = zstandard.ZstdCompressor(level=level)
    return len(c.compress(buf.tobytes())) * 8 / n_el


def shuffle_bytes(rows, w):
    return np.ascontiguousarray(rows.reshape(rows.shape[0], -1, w).transpose(2, 0, 1))


def analyze_tensor(rows, typ, kind, full):
    """rows: (n, rowbytes) uint8; typ: f16 | q8_0 | q4_0; kind 'K'/'V'; returns dict of bits/element."""
    r = {}
    if typ == "f16":
        n_el = rows.size // 2
        r.update({"f16." + k: v for k, v in model_f16(rows).items()})
        if kind == "K" and fmt.startswith("k4c"):
            sh, mh, q = k4c_encode(rows.view(np.float16))
            r.update({"k4c." + k: v for k, v in model_k4c(sh, mh, q).items()})
            r["k4c.f16_given_k4c"] = f16_given_k4c(rows.view(np.float16), sh, mh, q)
            if full:
                r["k4c.zstd3"] = zstd_bits(k4c_bytes(sh, mh, q), q.size)
        if full:
            r["f16.zstd3"] = zstd_bits(rows, n_el)
            r["f16.zstd3_shuf"] = zstd_bits(shuffle_bytes(rows, 2), n_el)
    else:
        A = 256 if typ == "q8_0" else 16
        d, q = (q8_0_split if typ == "q8_0" else q4_0_split)(rows)
        r.update({typ + "." + k: v for k, v in model_codes(d, q, A, None).items()})
        if full:
            r[typ + ".zstd3"] = zstd_bits(rows, q.size)
    return r


t0 = time.time()
for p in prefixes:
    agg = {"K": [], "V": []}
    for L in range(s.nlayer):
        for kind, T, typ in (("K", s.K[L], s.ktype), ("V", s.V[L], s.vtype)):
            if typ == "k4c":
                typ = "f16"  # session state stores k4c keys as f16
            agg[kind].append(analyze_tensor(np.array(T[:p]), typ, kind, p == prefixes[-1]))
    res["prefix"][p] = {kind: {k: float(np.mean([d[k] for d in lst if k in d])) for k in lst[0]} for kind, lst in agg.items()}
    res["prefix"][p]["per_layer"] = {kind: [{k: float(v) for k, v in d.items()} for d in lst] for kind, lst in agg.items()}
    print(path, p, f"{time.time() - t0:.0f}s", json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in res["prefix"][p].items() if k != "per_layer"}), flush=True)
json.dump(res, open(out, "w"), indent=1)
