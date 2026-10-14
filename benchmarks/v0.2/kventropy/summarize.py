"""summarize.py -> markdown tables of the KV entropy results (bits per element of the stored format, and ratios)."""
import glob, json, os, re
import numpy as np

R = os.environ.get("KV_RESULTS", "results")
RAW = {"f16": 16.0, "q8_0": 8.5, "q4_0": 4.5, "k4c": 5.0}


def load(m, c, f):
    p = f"{R}/{m}-{c}-{f}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def kvz_ratios():
    out = {}
    cur = None
    for line in open(f"{R}/kvz-ratios.txt"):
        if line.startswith("== "):
            cur = tuple(line.split()[1:])
        m = re.match(r"(K|V): raw [\d.]+ MiB -> kvz [\d.]+ MiB \(([\d.]+)x\).*zstd-3 ([\d.]+)x", line)
        if m and cur:
            out[cur + (m[1],)] = (float(m[2]), float(m[3]))
        m = re.match(r"total: ([\d.]+)x", line)
        if m and cur:
            out[cur + ("total",)] = (float(m[1]), None)
    return out


Z = kvz_ratios() if os.path.exists(f"{R}/kvz-ratios.txt") else {}
lines = []
P = lambda *a: lines.append(" ".join(str(x) for x in a))


def fmt(b, raw):
    return f"{b:.2f} ({raw / b:.2f}x)"


for m in ("1.7b", "8b", "8b32k"):
    for c in ("wiki-test", "code-eval"):
        if not any(load(m, c, f) for f in ("f16", "q8", "k4cq4", "k4cq8")):
            continue
        P(f"\n### Qwen3-{m.upper()} {c}\n")
        P("| cache | tokens | K raw | K order-0 bytes | K best static | K kvz | K zstd-3 | V raw | V order-0 | V best static | V kvz | V zstd-3 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in ("f16", "q8", "k4cq4", "k4cq8"):
            d = load(m, c, f)
            if not d:
                continue
            for p, r in sorted(d["prefix"].items(), key=lambda x: int(x[0])):
                K, V = r["K"], r["V"]
                row = [f, p]
                for kind, x in (("K", K), ("V", V)):
                    typ = d["ktype"] if kind == "K" else d["vtype"]
                    if typ in ("f16", "k4c"):
                        raw = 16.0
                        h0 = x["f16.planes_h0"]
                        best = min(x["f16.hi|ch+lo|hi"], x["f16.hi|ch+lo|ch"])
                    else:
                        raw = RAW[typ]
                        h0 = x[f"{typ}.q_h0"]
                        best = min(x[f"{typ}.q|ch"], x[f"{typ}.dq_t|ch"])
                    z = Z.get((m, c, f, p, kind))
                    kvz = f"{raw / z[0]:.2f} ({z[0]:.3f}x)" if z else "-"
                    zs = f"{z[1]:.3f}x" if z and z[1] else "-"
                    row += [f"{raw:g}", fmt(h0, raw), fmt(best, raw), kvz, zs]
                P("| " + " | ".join(row) + " |")
        # native k4c (in-memory format) and the conditional f16|k4c
        d = load(m, c, "k4cq4")
        if d:
            p = max(d["prefix"], key=int)
            K = d["prefix"][p]["K"]
            P(f"\nnative k4c keys ({p} tokens): codes+headers {K['k4c.q|ch']:.3f} bits of 5 ({5 / K['k4c.q|ch']:.3f}x), "
              f"codes {K['k4c.codes_only']:.3f} of 4, headers {K['k4c.header_bits_per_elem']:.3f} of 1, zstd-3 "
              f"{K.get('k4c.zstd3', float('nan')):.3f}; exact f16 given k4c: {K['k4c.f16_given_k4c']:.2f} bits")
        d = load(m, c, "f16")
        if d:
            p = max(d["prefix"], key=int)
            K, V = d["prefix"][p]["K"], d["prefix"][p]["V"]
            P(f"\nF16 ({p} tokens): bytewise H0 K {K['f16.byte_h0']:.2f} / V {V['f16.byte_h0']:.2f}; previous-token predictor "
              f"K {K['f16.delta_t']:.2f} / V {V['f16.delta_t']:.2f}; zstd-3 on byte-shuffled K {K.get('f16.zstd3_shuf', float('nan')):.2f} / "
              f"V {V.get('f16.zstd3_shuf', float('nan')):.2f}")

P("\n### Speculative KV coding (blog), predictor = same model with Q4_K_M weights, 8K tokens\n")
P("| model | corpus | F16 K | F16 V | Q8_0 K | Q8_0 V | k4c K (codes) | Q4_0 V |")
P("|---|---|---|---|---|---|---|---|")
for m in ("1.7b", "8b"):
    for c in ("wiki-test", "code-eval"):
        p = f"{R}/pred-{m}-{c}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cell = lambda k, raw: f"{d[k]:.2f} ({raw / d[k]:.2f}x)" if k in d else "-"
        P(f"| {m} | {c} | {cell('f16.K', 16)} | {cell('f16.V', 16)} | {cell('q8_0.K', 8.5)} | {cell('q8_0.V', 8.5)} | "
          f"{cell('k4c.K', 5)} | {cell('q4_0.V', 4.5)} |")
open(f"{R}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
