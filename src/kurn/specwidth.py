"""Cost-aware speculative verify width.

kurn's verify cost is a staircase, not a line. The KURN buffer type (integration/llama.cpp) runs one
activation column on the GEMV, 2 on the 2-column verify kernel, 3-4 on the 4-column kernel, 5-8 on the
8-column kernel, and wider batches as L2 row chunks x groups of 8 columns. A kernel compiled for `cols`
columns computes all of them (columns beyond M repeat column 0), so 3 and 5-7 columns pay for 4 and 8.
A drafter that ignores the staircase pays for padded columns and for the extra pass at 9+.

    kurn specwidth kernels [--formats q8_0,q4_0] [--shapes K:N:count,...] [--head] [--threads T] [--out costs.csv]
        GEMV + verify 2/4/8 at every width they serve, with the buffer type's kernel configs, on a
        model's matmul shapes (default: Qwen3-8B, 36 layers; --head adds the output head), DRAM-streaming regime
    kurn specwidth show TABLE                     print a whole-forward cost table and the widths it favours
    kurn specwidth simulate TABLE TRACE [...]     replay greedy acceptance traces: fixed widths vs the policy

The whole-forward table (target and draft forward time for M tokens, M = 1..Mmax) is measured inside
llama.cpp by `kurn-spec-calib` (integration/llama.cpp/spec-width). Its text format, shared with the C++
policy (`kurn-spec-width.h`):

    # comment
    verify M MS        target forward, M tokens, logits for all M
    draft M MS         draft forward, M tokens

WidthPolicy picks the draft length k (verify width M = k + 1) that maximises expected accepted tokens minus
lambda * time, where lambda is the running throughput in tokens/ms (Dinkelbach's method for maximising
E[tokens] / E[time] over a renewal process). Per-token acceptance is estimated online from verification
feedback, globally and per draft-confidence bin. It is used three ways, each step:
  - n_cap():           draft length cap from the global acceptance rate (drafters without probabilities)
  - keep_drafting(p):  after each drafted token: is drafting another one worth its draft time and the
                       verify-cost step it may cross?
  - truncate(p):       after drafting: the prefix whose verify width maximises the objective
`kurn-spec-width.h` is a line-by-line twin; tests/test_specwidth.py checks that they agree.
"""

import argparse
import math
import os
import statistics
import sys

# --------------------------------------------------------------------------- kernel staircase
VFY_COLS = (2, 4, 8)
VFY_MAX = 8
# The buffer type's kernel keys: DEFAULT_KEYS + TUNED_KEYS of integration/llama.cpp/gen_ggml_sources.py
# (keep in sync), so the kernel table times the kernels llama.cpp runs.
BUFT_DEFAULT_KEYS = {"layout": "i16", "rows": 2, "prefetch": 0}
BUFT_TUNED_KEYS = {
    "q8_0": {"rows": 8},
    "q4_0": {"unpack": "pair", "rows": 4},
    "q4_K": {"unpack": "pair", "correction": "dpmin", "rows": 4},
    "iq4_nl": {"unpack": "perm", "rows": 4},
}
# One Qwen3-8B decoder layer (q, k, v, o, gate, up, down as K:N) plus the output head (151936 rows).
QWEN3_8B_SHAPES = ((4096, 4096, 2), (4096, 1024, 2), (4096, 12288, 2), (12288, 4096, 1))
QWEN3_8B_HEAD = (4096, 151936, 1)


def kernel_cols(m):
    """Columns of the kernel the buffer type runs for an m-column group (m <= 8): 1 = GEMV."""
    if m < 1 or m > VFY_MAX:
        raise ValueError(f"group width {m} outside 1..{VFY_MAX}")
    return 1 if m == 1 else next(c for c in VFY_COLS if c >= m)


def kernel_passes(M):
    """Kernel calls the buffer type makes for M columns: groups of 8, then the remainder."""
    out = []
    while M > 0:
        g = min(VFY_MAX, M)
        out.append(kernel_cols(g))
        M -= g
    return out


def buft_configs(fmt, threads):
    """{cols: resolved config} for the GEMV (cols 1) and the 2/4/8-column verify kernels."""
    from .spec import resolve

    keys = {**BUFT_DEFAULT_KEYS, **BUFT_TUNED_KEYS.get(fmt, {})}
    g = resolve({"op": "gemv", "weights": fmt, "target": "avx512_vnni", "threads": threads, **keys})
    out = {1: g}
    vkeys = {k: g[k] for k in ("unpack", "correction", "scales", "accum", "ilv") if k in g}
    for cols in VFY_COLS:
        out[cols] = resolve(
            {"op": "verify", "weights": fmt, "target": "avx512_vnni", "layout": g["layout"], "rows": 1, "cols": cols,
             "prefetch": g["prefetch"], "threads": threads, **vkeys}
        )  # fmt: skip
    return out


def measure_kernels(fmt, shapes, threads=None, regime="cold", secs=0.5, reps=3, widths=range(1, VFY_MAX + 1), log=print):
    """Median us per call of the kernel serving each width, per shape. Reps are interleaved over
    (shape, width) so drift spreads evenly. Returns rows {fmt, K, N, count, M, cols, us, GBps, relerr}."""
    from .harness import bench
    from .toolchain import build

    threads = threads or os.cpu_count() or 1
    cfgs = buft_configs(fmt, threads)
    sos = {cols: build(c) for cols, c in cfgs.items()}
    samples = {}
    for rep in range(reps):
        for K, N, count in shapes:
            for M in widths:
                cols = kernel_cols(M)
                extra = ["--K", str(K), "--N", str(N)] + (["--M", str(M)] if cols > 1 else [])
                row = bench(sos[cols], cfgs[cols], regime, secs, extra)
                if row["check"] == "FAIL":
                    raise RuntimeError(f"{fmt} K={K} N={N} M={M}: kernel failed its correctness check")
                samples.setdefault((K, N, count, M), []).append(row)
        log(f"{fmt}: rep {rep + 1}/{reps} done")
    out = []
    for (K, N, count, M), rows in samples.items():
        us = statistics.median(r["us"] for r in rows)
        out.append({
            "fmt": fmt, "K": K, "N": N, "count": count, "M": M, "cols": kernel_cols(M), "us": round(us, 2),
            "GBps": round(statistics.median(r["GBps"] for r in rows), 1),
            "spread": round((max(r["us"] for r in rows) - min(r["us"] for r in rows)) / us, 3),
            "relerr": max(r["relerr"] for r in rows),
        })  # fmt: skip
    return out


def parse_shapes(s):
    out = []
    for part in s.split(","):
        f = [int(x) for x in part.split(":")]
        out.append((f[0], f[1], f[2] if len(f) > 2 else 1))
    return tuple(out)


# --------------------------------------------------------------------------- cost tables
class CostTable:
    """Whole-forward cost: verify_ms[M - 1] = target forward with M tokens; draft_ms[M - 1] = draft forward."""

    def __init__(self, verify_ms, draft_ms=(), meta=None):
        if not verify_ms or any(not (v > 0 and math.isfinite(v)) for v in verify_ms):
            raise ValueError("verify costs must be positive and finite, starting at M = 1")
        self.verify_ms = list(verify_ms)
        self.draft_ms = list(draft_ms)
        self.meta = meta or []

    @property
    def m_max(self):
        return len(self.verify_ms)

    def verify(self, M):
        return self.verify_ms[M - 1]

    def draft_step(self):
        """Marginal draft time per drafted token: one single-token draft forward plus the growth of
        the draft's re-decode of the verify batch per extra column."""
        if not self.draft_ms:
            return 0.0
        d = self.draft_ms
        slope = (d[-1] - d[0]) / (len(d) - 1) if len(d) > 1 else 0.0
        return d[0] + max(0.0, slope)

    @classmethod
    def load(cls, path):
        v, d, meta = {}, {}, []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    meta.append(line[1:].strip())
                    continue
                kind, m, ms = line.split()[:3]
                (v if kind == "verify" else d if kind == "draft" else {})[int(m)] = float(ms)
        for name, t in (("verify", v), ("draft", d)):
            if t and sorted(t) != list(range(1, len(t) + 1)):
                raise ValueError(f"{path}: {name} rows must cover M = 1..n without gaps")
        return cls([v[m] for m in sorted(v)], [d[m] for m in sorted(d)], meta)

    def save(self, path):
        with open(path, "w") as fh:
            for m in self.meta:
                fh.write(f"# {m}\n")
            for i, ms in enumerate(self.verify_ms):
                fh.write(f"verify {i + 1} {ms:.4f}\n")
            for i, ms in enumerate(self.draft_ms):
                fh.write(f"draft {i + 1} {ms:.4f}\n")


# --------------------------------------------------------------------------- policy
# Draft-confidence bins (top-1 probability of the draft's sampler). Most greedy drafts sit near 1.
BIN_EDGES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0)


def _bin(p):
    for b in range(len(BIN_EDGES) - 2):
        if p < BIN_EDGES[b + 1]:
            return b
    return len(BIN_EDGES) - 2


class WidthPolicy:
    """Online verify-width policy (see the module docstring). Mirrors kurn-spec-width.h exactly."""

    def __init__(self, table, k_max=None, alpha0=0.6, prior=4.0, decay=0.98, lam_decay=0.9, probe_every=16, use_conf=True):
        self.t = table
        self.k_max = min(k_max if k_max is not None else table.m_max - 1, table.m_max - 1)
        self.draft_ms = table.draft_step()
        self.alpha0, self.prior, self.decay, self.lam_decay = alpha0, prior, decay, lam_decay
        self.probe_every, self.use_conf = probe_every, use_conf
        nb = len(BIN_EDGES) - 1
        self.bs, self.bn = [0.0] * nb, [0.0] * nb
        self.gs = self.gn = 0.0
        self.lam_num = self.lam_den = 0.0
        self.idle = 0

    # -- estimates
    def alpha(self):
        return (self.gs + self.prior * self.alpha0) / (self.gn + self.prior)

    def acc(self, p):
        if p is None or not self.use_conf:
            return self.alpha()
        b = _bin(p)
        mid = 0.5 * (BIN_EDGES[b] + BIN_EDGES[b + 1])
        return (self.bs[b] + self.prior * mid) / (self.bn[b] + self.prior)

    def lam(self):
        return self.lam_num / self.lam_den if self.lam_den > 0 else 1.0 / self.t.verify(1)

    # -- objective: U(k) = G(k) - lam * verify(k + 1), G(k) = sum_{i<=k} prod_{j<=i} acc(p_j)
    def _values(self, probs):
        lam, q, g = self.lam(), 1.0, 0.0
        u = [-lam * self.t.verify(1)]
        for p in probs:
            q *= self.acc(p)
            g += q
            u.append(g - lam * self.t.verify(len(u) + 1))
        return u, q, g

    def _best_future(self, j, q, g):
        """Best value of stopping at some k' > j, future tokens accepted at the global rate."""
        lam, a = self.lam(), self.alpha()
        best, s, r = -math.inf, 0.0, 1.0
        for k in range(j + 1, self.k_max + 1):
            r *= a
            s += r
            best = max(best, g + q * s - lam * (self.t.verify(k + 1) + (k - j) * self.draft_ms))
        return best

    def keep_drafting(self, probs):
        j = len(probs)
        if j >= self.k_max:
            return False
        u, q, g = self._values(probs)
        return self._best_future(j, q, g) > max(u) + 1e-9

    def truncate(self, probs):
        u, _, _ = self._values(probs)
        best = 0
        for k in range(1, len(u)):
            if u[k] > u[best] + 1e-9:
                best = k
        return best

    def n_cap(self):
        """Draft length from the global acceptance rate alone (0: decode without drafting this step)."""
        best_k, best = 0, -self.lam() * self.t.verify(1)
        lam, a, s, r = self.lam(), self.alpha(), 0.0, 1.0
        for k in range(1, self.k_max + 1):
            r *= a
            s += r
            v = s - lam * (self.t.verify(k + 1) + k * self.draft_ms)
            if v > best + 1e-9:
                best_k, best = k, v
        if best_k == 0 and self.idle + 1 >= self.probe_every:
            return 1
        return best_k

    # -- feedback
    def observe(self, probs, n_accepted, step_ms, n_tokens):
        """probs: draft-confidence of the verified draft tokens (None entries allowed), n_accepted of them
        accepted; the step took step_ms and committed n_tokens."""
        k = len(probs)
        self.idle = 0 if k else self.idle + 1
        d = self.decay
        if k:
            self.bs = [x * d for x in self.bs]
            self.bn = [x * d for x in self.bn]
            self.gs *= d
            self.gn *= d
        for i in range(min(k, n_accepted + 1)):
            ok = 1.0 if i < n_accepted else 0.0
            self.gs += ok
            self.gn += 1.0
            if probs[i] is not None:
                b = _bin(probs[i])
                self.bs[b] += ok
                self.bn[b] += 1.0
        self.lam_num = self.lam_decay * self.lam_num + n_tokens
        self.lam_den = self.lam_decay * self.lam_den + step_ms


# --------------------------------------------------------------------------- trace replay
def load_trace(path):
    """Greedy acceptance trace from kurn-spec-calib: one line per generated position c,
    `match p` = does the draft's top-1 given the true prefix equal the target's token, and its probability."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                m, p = line.split()[:2]
                out.append((m == "1", float(p)))
    return out


def step_time(table, k):
    """Modelled time of one step with k drafts: target verify of k + 1 tokens, k single-token draft forwards
    (id_last, then k - 1 drafted tokens) and the draft's re-decode of the verified batch."""
    t = table.verify(k + 1)
    if table.draft_ms:
        t += table.draft_ms[k] if k + 1 <= len(table.draft_ms) else table.draft_ms[-1]
        t += k * table.draft_ms[0]
    return t


def simulate(table, trace, policy="fixed", width=4, p_min=0.0, **kw):
    """Replay a greedy trace. policy: "fixed" (draft `width` tokens, stop early below p_min),
    "cap" (WidthPolicy.n_cap only), "policy" (n_cap + keep_drafting + truncate).
    Returns {tokens, ms, tok_s, steps, drafted, accepted, widths: {M: steps}}."""
    pol = WidthPolicy(table, **kw) if policy != "fixed" else None
    n = len(trace)
    c = 0
    ms = drafted = accepted = steps = 0
    widths = {}
    while c < n:
        if pol is None:
            kmax = min(width, table.m_max - 1)
        else:
            kmax = pol.n_cap() if policy == "cap" else pol.k_max
        probs, extra_fwd = [], 0
        while len(probs) < kmax and c + len(probs) < n:
            p = trace[c + len(probs)][1]
            if pol is None and p < p_min:
                extra_fwd = 1 if probs else 0  # the last drafted token was forwarded to produce this one
                break
            probs.append(p)
            if policy == "policy" and not pol.keep_drafting(probs):
                break
        k_drafted = len(probs)
        if policy == "policy":
            probs = probs[: pol.truncate(probs)]
        k = len(probs)
        a = 0
        while a < k and trace[c + a][0]:
            a += 1
        dt = step_time(table, k)
        if table.draft_ms:  # drafted, then truncated: their draft forwards were still paid
            dt += (k_drafted - k + extra_fwd) * table.draft_ms[0]
        got = min(a + 1, n - c)
        if pol is not None:
            pol.observe(probs, a, dt, got)
        c += got
        ms += dt
        drafted += k
        accepted += a
        steps += 1
        widths[k + 1] = widths.get(k + 1, 0) + 1
    return {"tokens": c, "ms": ms, "tok_s": 1000.0 * c / ms, "steps": steps, "drafted": drafted, "accepted": accepted,
            "widths": dict(sorted(widths.items()))}  # fmt: skip


# --------------------------------------------------------------------------- CLI
def _cmd_kernels(a):
    import csv

    if a.from_csv:
        rows = [{**r, "K": int(r["K"]), "N": int(r["N"]), "count": int(r["count"]), "M": int(r["M"]), "us": float(r["us"])}
                for r in csv.DictReader(open(a.from_csv))]  # fmt: skip
    else:
        shapes = parse_shapes(a.shapes) if a.shapes else QWEN3_8B_SHAPES + ((QWEN3_8B_HEAD,) if a.head else ())
        rows = []
        for fmt in a.formats.split(","):
            rows += measure_kernels(fmt, shapes, a.threads, a.regime, a.secs, a.reps, log=lambda m: print(m, file=sys.stderr))
    fields = ["fmt", "K", "N", "count", "M", "cols", "us", "GBps", "spread", "relerr"]
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    layers = 1 if a.shapes else a.layers
    for fmt in dict.fromkeys(r["fmt"] for r in rows):
        tot = {}
        for r in rows:
            if r["fmt"] == fmt:
                n = r["count"] * (1 if (r["K"], r["N"]) == QWEN3_8B_HEAD[:2] else layers)
                tot[r["M"]] = tot.get(r["M"], 0.0) + r["us"] * n
        heads = any((r["K"], r["N"]) == QWEN3_8B_HEAD[:2] for r in rows if r["fmt"] == fmt)
        what = "shape set" if a.shapes else f"forward: {layers} layers{' + head' if heads else ''}"
        print(f"{fmt}: matmul time per {what} (ms) by verify width M")
        print("  M  kernel     ms    vs M=1   ms/token")
        for M in sorted(tot):
            name = "gemv" if M == 1 else f"vfy{kernel_cols(M)}"
            print(f"  {M}  {name:6s} {tot[M] / 1e3:7.2f}  {tot[M] / tot[1]:6.2f}x  {tot[M] / M / 1e3:7.2f}")
    return 0


def _cmd_show(a):
    t = CostTable.load(a.table)
    for m in t.meta:
        print(f"# {m}")
    print(f"marginal draft cost {t.draft_step():.2f} ms/token")
    print("   M   verify ms   x M=1   ms/token  kernels")
    for M in range(1, t.m_max + 1):
        v = t.verify(M)
        print(f"  {M:2d}  {v:9.2f}  {v / t.verify(1):6.2f}  {v / M:8.2f}  {'+'.join(map(str, kernel_passes(M)))}")
    print("\nbest draft length by acceptance rate (geometric acceptance, lambda iterated to its fixed point):")
    for alpha in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        k, rate = best_static(t, alpha, a.k_max)
        print(f"  alpha {alpha:.2f}: k = {k:2d} (M = {k + 1:2d}), {1000 * rate:6.2f} tok/s")
    return 0


def best_static(table, alpha, k_max=None):
    """Draft length maximising expected tokens / step time for i.i.d. acceptance alpha (no policy state)."""
    k_max = min(k_max or table.m_max - 1, table.m_max - 1)
    best_k, best = 0, 0.0
    for k in range(0, k_max + 1):
        tokens = sum(alpha**i for i in range(k + 1))
        rate = tokens / step_time(table, k)
        if rate > best:
            best_k, best = k, rate
    return best_k, best


def _cmd_simulate(a):
    t = CostTable.load(a.table)
    traces = [(p, load_trace(p)) for p in a.trace]
    runs = [("fixed", w, 0.0) for w in a.widths] + [("fixed", a.pmin_width, pm) for pm in a.p_min]
    runs += [("cap", None, 0.0), ("policy", None, 0.0)]
    print(f"{'config':18s} " + " ".join(f"{os.path.basename(p)[:14]:>14s}" for p, _ in traces) + "      mean")
    for pol, w, pm in runs:
        name = f"fixed k={w}" + (f" p>{pm}" if pm else "") if pol == "fixed" else f"{pol} (k<={a.k_max})"
        res = [simulate(t, tr, pol, w, pm, **({"k_max": a.k_max} if pol != "fixed" else {})) for _, tr in traces]
        rates = [r["tok_s"] for r in res]
        print(f"{name:18s} " + " ".join(f"{x:14.2f}" for x in rates) + f"  {statistics.mean(rates):8.2f}")
        if a.verbose:
            for (p, _), r in zip(traces, res):
                print(f"    {os.path.basename(p)}: steps {r['steps']} drafted {r['drafted']} accepted {r['accepted']} widths {r['widths']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kurn specwidth", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kernels", help="kernel cost per verify width on a model's matmul shapes")
    p.add_argument("--formats", default="q8_0")
    p.add_argument("--shapes", help="K:N:count,... (default: the Qwen3-8B decoder layer shapes)")
    p.add_argument("--layers", type=int, default=36, help="decoder layers the default shapes are counted for (Qwen3-8B: 36)")
    p.add_argument("--head", action="store_true", help="add the Qwen3-8B output head to the default shapes")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--regime", default="cold", choices=["hot", "cold"])
    p.add_argument("--secs", type=float, default=0.5)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--from-csv", help="summarise a previous --out instead of measuring")
    p.set_defaults(fn=_cmd_kernels)
    p = sub.add_parser("show", help="print a whole-forward cost table")
    p.add_argument("table")
    p.add_argument("--k-max", type=int)
    p.set_defaults(fn=_cmd_show)
    p = sub.add_parser("simulate", help="replay greedy traces: fixed widths vs the policy")
    p.add_argument("table")
    p.add_argument("trace", nargs="+")
    p.add_argument("--widths", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3, 4, 7, 8, 15])
    p.add_argument("--p-min", type=lambda s: [float(x) for x in s.split(",")], default=[], help="fixed-width runs with a p_min cutoff")
    p.add_argument("--pmin-width", type=int, default=16)
    p.add_argument("--k-max", type=int, default=15)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_cmd_simulate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
