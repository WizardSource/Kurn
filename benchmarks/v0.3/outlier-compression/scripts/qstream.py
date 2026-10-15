"""Layer-streaming Qwen3 forward (fp32 compute, bf16 safetensors) for kurn-internals prototypes.

One weight stream feeds many numerics configs at once: every config keeps its own hidden states, and each
layer's weights are loaded (and quantized per weight scheme) once. Fits Qwen3-8B in ~6 GB.

  python qstream.py MODEL_DIR --mode stats  --out stats.json
  python qstream.py MODEL_DIR --mode kl     --stats stats.json --out kl.json
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

torch.set_grad_enabled(False)


class Weights:
    def __init__(self, d):
        idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"] if os.path.exists(
            f"{d}/model.safetensors.index.json") else None
        self.d, self.idx, self.h = d, idx, {}

    def get(self, name):
        f = self.idx[name] if self.idx else "model.safetensors"
        if f not in self.h:
            self.h[f] = safe_open(f"{self.d}/{f}", framework="pt")
        return self.h[f].get_tensor(name).float()


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(T, hd, theta, scaling=None):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    if scaling and scaling.get("rope_type") == "llama3":
        f, lo, hi, old = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"], scaling[
            "original_max_position_embeddings"]
        wl = 2 * math.pi / inv
        smooth = (old / wl - lo) / (hi - lo)
        scaled = torch.where(wl > old / lo, inv / f, inv)
        mid = (wl <= old / lo) & (wl >= old / hi)
        inv = torch.where(mid, (1 - smooth) * inv / f + smooth * inv, scaled)
    f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().float(), emb.sin().float()


def rot(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


# ---------------------------------------------------------------- numerics
def q_tok(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    return (x / s).round().clamp(-127, 127) * s


def q_grp(x, g):
    sh = x.shape
    y = x.reshape(*sh[:-1], sh[-1] // g, g)
    return q_tok(y).reshape(sh)


def bf16(x):
    return x.to(torch.bfloat16).float()


def wq_row(W):
    return q_tok(W)


def wq_g32(W):
    return q_grp(W, 32)


_HAD = {}


def hblock(K):
    b = 1
    while K % (b * 2) == 0 and b < 4096:
        b *= 2
    return b


def hadamard(x):
    """orthonormal block Hadamard along the last dim (largest power-of-2 block dividing K, <= 4096); self-inverse"""
    K = x.shape[-1]
    b = hblock(K)
    if b not in _HAD:
        H = torch.ones(1, 1)
        while H.shape[0] < b:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD[b] = H / math.sqrt(b)
    sh = x.shape
    return (x.reshape(*sh[:-1], K // b, b) @ _HAD[b]).reshape(sh)


class Act:
    """activation numerics at one linear-input site. spec: none|bf16|g32|g256|tok, +out (top-k channels kept
    in bf16), +shift (calibration channel mean subtracted before quantization, added back exactly), +rot (block
    Hadamard before quantization, i.e. weights stored rotated)"""

    def __init__(self, spec, k=16):
        parts = spec.split("+")
        self.base, self.out, self.shift, self.k = parts[0], "out" in parts, "shift" in parts, k
        self.rot = "rot" in parts

    def __call__(self, x, site):
        if self.base == "none":
            return x
        if self.base == "bf16":
            return bf16(x)
        if self.rot:
            return hadamard(q_tok(hadamard(x)) if self.base == "tok" else q_grp(hadamard(x), int(self.base[1:])))
        mu = site["mean"] if self.shift else None
        idx = site["top"][: self.k] if self.out else None
        y = x - mu if mu is not None else x
        if idx is not None:
            keep = y[..., idx].clone()
            y = y.clone()
            y[..., idx] = 0
        if self.base == "tok":
            y = q_tok(y)
        elif self.base.startswith("g"):
            y = q_grp(y, int(self.base[1:]))
        if idx is not None:
            y[..., idx] = bf16(keep)
        return y + mu if mu is not None else y


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, d):
        self.cfg = json.load(open(f"{d}/config.json"))
        self.W = Weights(d)
        c = self.cfg
        self.L, self.H, self.KV = c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"]
        self.hd = c.get("head_dim") or c["hidden_size"] // self.H
        self.qknorm = c.get("model_type") == "qwen3"
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")

    def layer(self, i):
        p = f"model.layers.{i}."
        g = self.W.get
        names = dict(
            ln1="input_layernorm.weight", ln2="post_attention_layernorm.weight", q="self_attn.q_proj.weight",
            k="self_attn.k_proj.weight", v="self_attn.v_proj.weight", o="self_attn.o_proj.weight",
            gate="mlp.gate_proj.weight", up="mlp.up_proj.weight", down="mlp.down_proj.weight")
        if self.qknorm:
            names.update(qn="self_attn.q_norm.weight", kn="self_attn.k_norm.weight")
        return {k: g(p + v) for k, v in names.items()}

    def block(self, x, w, cs, act, sites, hook=None, li=0):
        """x: [B,T,D]. act(xin, sitename) -> numerics-applied input. w: dict of (possibly quantized) weights."""
        B, T, D = x.shape
        cos, sin = cs
        h = rms(x, w["ln1"], self.eps)
        if hook: hook("qkv", li, h)
        hq = act(h, "qkv")
        lin = lambda xin, n: xin @ w[n].T + w["b_" + n] if "b_" + n in w else xin @ w[n].T
        q = lin(hq, "q").view(B, T, self.H, self.hd)
        k = lin(hq, "k").view(B, T, self.KV, self.hd)
        v = lin(hq, "v").view(B, T, self.KV, self.hd)
        if self.qknorm:
            q, k = rms(q, w["qn"], self.eps), rms(k, w["kn"], self.eps)
        if getattr(self, "kvq", None) is not None:  # KV numerics on pre-RoPE K (k4c stores K pre-RoPE) and V
            k, v = self.kvq(k, v, li)
        q = rot(q.transpose(1, 2), cos, sin)
        k = rot(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)
        rep = self.H // self.KV
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
        if hook: hook("o", li, a)
        x = x + lin(act(a, "o"), "o")
        h = rms(x, w["ln2"], self.eps)
        if hook: hook("gu", li, h)
        hg = act(h, "gu")
        m = F.silu(lin(hg, "gate")) * lin(hg, "up")
        if hook: hook("down", li, m)
        x = x + lin(act(m, "down"), "down")
        return x

    def head(self):
        g = self.W.get
        name = "model.embed_tokens.weight" if self.cfg.get("tie_word_embeddings") else "lm_head.weight"
        return g("model.norm.weight"), g(name)


TEXT = "/tmp/wiki.txt"


def chunks(tok, n, T, skip=0):
    ids = tok.encode(open(TEXT).read()[: (skip + n) * T * 8]).ids
    out = [ids[(skip + i) * T:(skip + i + 1) * T] for i in range(n)]
    return torch.tensor(out)


# ---------------------------------------------------------------- stats pass
def run_stats(m, args):
    ids = chunks(m.tok, args.calib, args.T)
    emb = m.W.get("model.embed_tokens.weight")
    x = emb[ids]
    del emb
    cs = rope(args.T, m.hd, m.theta, m.cfg.get('rope_scaling'))
    D = x.shape[-1]
    out = {"model": args.model, "tokens": int(ids.numel()), "layers": []}
    sites = {}

    def hook(name, li, h):
        f = h.reshape(-1, h.shape[-1])
        nf = h[:, 1:].reshape(-1, h.shape[-1])  # non-first tokens
        en = nf.pow(2).mean(0)
        top = torch.argsort(en, descending=True)
        tokr = (f.abs().amax(-1) / f.pow(2).mean(-1).sqrt())
        rel = lambda y: ((y - f).norm() / f.norm()).item()
        s = {"mean": nf.mean(0), "top": top[:64], "ms": f.pow(2).mean(0)}
        A = lambda spec: rel(Act(spec)(f, s))
        sites[(name, li)] = s
        out["layers"][li]["sites"][name] = {
            "top16_energy_share": (en[top[:16]].sum() / en.sum()).item(),
            "top16": top[:16].tolist(),
            "absmax_over_rms_median": tokr.median().item(),
            "absmax_over_rms_first_tok": tokr.view(h.shape[0], -1)[:, 0].mean().item(),
            "max_abs": f.abs().max().item(),
            "relerr_g32": A("g32"), "relerr_tok": A("tok"), "relerr_tok_out16": A("tok+out"),
            "relerr_tok_shift": A("tok+shift"), "relerr_tok_shift_out16": A("tok+shift+out"), "relerr_bf16": A("bf16"),
        }

    def resid_stats(li, x):
        f = x[:, 1:].reshape(-1, D)
        first = x[:, 0]
        en = f.pow(2).sum(0)
        tot = en.sum()
        top = torch.argsort(en, descending=True)[:8]
        en_all = x.reshape(-1, D).pow(2).sum(0)
        top_all = torch.argsort(en_all, descending=True)[:8]
        t5 = top[:5]
        mu, sd = f[:, t5].mean(0), f[:, t5].std(0)
        return {
            "top5_dims_nonfirst": t5.tolist(), "top5_energy_share_nonfirst": (en[t5].sum() / tot).item(),
            "top5_dims_all": top_all[:5].tolist(),
            "top5_energy_share_all": (en_all[top_all[:5]].sum() / en_all.sum()).item(),
            "top5_mean": mu.tolist(), "top5_std": sd.tolist(),
            "top5_cv": (sd / mu.abs().clamp_min(1e-9)).tolist(),
            "rms_nonfirst": f.pow(2).mean().sqrt().item(),
            "first_tok_absmax": first.abs().max().item(),
            "first_tok_argmax": first.abs().amax(0).argmax().item(),
            "first_tok_energy_x_nonfirst": (first.pow(2).sum(-1).mean() / f.pow(2).sum(-1).mean()).item(),
            "max_abs_nonfirst": f.abs().max().item(),
            "_mean_full": f.mean(0), "_mean_first": first.mean(0),
        }

    means = []
    for li in range(m.L):
        t0 = time.time()
        w = m.layer(li)
        out["layers"].append({"sites": {}})
        x = m.block(x, w, cs, lambda h, s: h, None, hook, li)
        r = resid_stats(li, x)
        means.append({"mean": r.pop("_mean_full"), "first": r.pop("_mean_first")})
        out["layers"][li]["resid"] = r
        print(f"layer {li} {time.time() - t0:.1f}s top5 share {r['top5_energy_share_nonfirst']:.3f} "
              f"dims {r['top5_dims_nonfirst']} first-tok absmax {r['first_tok_absmax']:.0f}", flush=True)
    nw, hw = m.head()
    h = rms(x, nw, m.eps)
    out["layers"].append({"sites": {}})
    hook("head", m.L, h)
    json.dump(out, open(args.out, "w"), indent=1)
    torch.save({"sites": {f"{k[0]}:{k[1]}": v for k, v in sites.items()}, "resid": means}, args.out + ".pt")


# ---------------------------------------------------------------- KL pass
def wq_row_mse(W):
    """per-row int8 with the clipping scale (of 0.70..1.00 x absmax) that minimizes the row's squared error"""
    amax = W.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    best, best_e = None, None
    for c in torch.linspace(0.70, 1.0, 13):
        s = amax * c / 127
        q = (W / s).round().clamp(-127, 127) * s
        e = (q - W).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_e = q, e
        else:
            m = e < best_e
            best = torch.where(m, q, best)
            best_e = torch.where(m, e, best_e)
    return best


WSCHEMES = {"bf16": lambda W: W, "row": wq_row, "g32": wq_g32,
            "rowrot": lambda W: hadamard(wq_row(hadamard(W))),
            "rowrotmse": lambda W: hadamard(wq_row_mse(hadamard(W)))}
WSITE = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gu", "up": "gu", "down": "down", "head": "head"}


def per_site(spec):
    """'tok+out;down=bf16' -> {'*': 'tok+out', 'down': 'bf16'}"""
    d = {}
    for p in spec.split(";"):
        k, v = p.split("=") if "=" in p else ("*", p)
        d[k] = v
    return d


def site_spec(d, site):
    return d.get(site, d["*"])


CONFIGS = [  # name, activation spec, weight scheme (';site=' overrides)
    ("act-bf16 (kurn prefill numerics, bf16 W)", "bf16", "bf16"),
    ("A8 g32 (kurn decode act), bf16 W", "g32", "bf16"),
    ("A8 per-token, bf16 W", "tok", "bf16"),
    ("A8 per-token +out16, bf16 W", "tok+out", "bf16"),
    ("A8 per-token +shift, bf16 W", "tok+shift", "bf16"),
    ("A8 per-token +shift+out16, bf16 W", "tok+shift+out", "bf16"),
    ("A8 per-token +Hadamard, bf16 W", "tok+rot", "bf16"),
    ("A8 g256, bf16 W", "g256", "bf16"),
    ("act-bf16, W8 g32 (= kurn Q8_0 prefill)", "bf16", "g32"),
    ("A8 g32, W8 g32 (= kurn Q8_0 decode)", "g32", "g32"),
    ("A8 per-token, W8 per-row (fast AMX-INT8)", "tok", "row"),
    ("A8 per-token +out16, W8 per-row", "tok+out", "row"),
    ("A8 per-token +shift+out16, W8 per-row", "tok+shift+out", "row"),
    ("A8 per-token +out4, W8 per-row", "tok+out4", "row"),
    ("A8 per-token +Hadamard, W8 per-row (rotated)", "tok+rot", "rowrot"),
    ("mixed: A8 tok+out16 W8 row; down: bf16 act, W8 g32", "tok+out;down=bf16", "row;down=g32"),
    ("mixed: A8 tok+out16 W8 row; down: Hadamard A8/W8", "tok+out;down=tok+rot", "row;down=rowrot"),
    ("R8: A8 per-token +Hadamard, W8 per-row MSE-clipped (rotated)", "tok+rot", "rowrotmse"),
    ("W8 per-row rotated, fp32 act", "none", "rowrot"),
    ("W8 per-row MSE-clipped rotated, fp32 act", "none", "rowrotmse"),
]

RESID_CONFIGS = ["dark5->const", "dark5->zero", "dark5->const (all tokens)", "giant (>50x rms)->zero"]


def run_kl(m, args):
    st = torch.load(args.stats + ".pt")
    sj = json.load(open(args.stats))
    S = st["sites"]
    ids = chunks(m.tok, args.eval, args.T, skip=args.calib)
    cs = rope(args.T, m.hd, m.theta, m.cfg.get('rope_scaling'))
    emb = m.W.get("model.embed_tokens.weight")
    x0 = emb[ids]
    del emb
    D = x0.shape[-1]
    mid = m.L // 2
    dark = torch.tensor(sj["layers"][mid]["resid"]["top5_dims_nonfirst"])
    cfgs = [("ref (bf16 W, fp32 act)", "none", "bf16")] + list(CONFIGS) + [(r, "none", "bf16") for r in RESID_CONFIGS]
    if args.only:
        keep = set(args.only.split(","))
        cfgs = [c for i, c in enumerate(cfgs) if i == 0 or str(i) in keep]
    xs = [x0.clone() for _ in cfgs]
    del x0

    def one(spec):
        k = 16
        if "out4" in spec:
            spec, k = spec.replace("out4", "out"), 4
        return Act(spec, k)

    def mkact(spec, li):
        d = per_site(spec)
        acts = {s: one(site_spec(d, s)) for s in ("qkv", "o", "gu", "down", "head")}
        return lambda h, name: acts[name](h, S[f"{name}:{li}"])

    wcache = {}

    def wget(w, k, sch):
        key = (k, sch)
        if key not in wcache:
            wcache[key] = WSCHEMES[sch](w[k])
        return wcache[key]

    for li in range(m.L):
        t0 = time.time()
        w = m.layer(li)
        wcache.clear()
        for ci, (name, aspec, ws) in enumerate(cfgs):
            wd = per_site(ws)
            ww = {k: (wget(w, k, site_spec(wd, WSITE[k])) if k in WSITE else v) for k, v in w.items()}
            xs[ci] = m.block(xs[ci], ww, cs, mkact(aspec, li), None, None, li)
            if name.startswith("dark5") and li < m.L - 1:
                mu = st["resid"][li]["mean"][dark]
                sl = slice(None) if "all tokens" in name else slice(1, None)
                xs[ci][:, sl, dark] = 0 if "zero" in name else mu
            if name.startswith("giant") and li < m.L - 1:
                thr = 50 * sj["layers"][li]["resid"]["rms_nonfirst"]
                xs[ci] = torch.where(xs[ci].abs() > thr, torch.zeros_like(xs[ci]), xs[ci])
        print(f"layer {li} {time.time() - t0:.1f}s", flush=True)
        del w
    wcache.clear()
    nw, hw = m.head()
    res = {n: {"kl": 0.0, "top1": 0.0, "nll": 0.0} for n, _, _ in cfgs}
    B, T = ids.shape
    tgt = ids[:, 1:]
    n = 0
    for b in range(B):
        lp_ref = None
        for ci, (name, aspec, ws) in enumerate(cfgs):
            # output head kept at bf16 W / fp32 act for every config: prefill only needs the last token's logits
            lp = F.log_softmax(rms(xs[ci][b, :-1], nw, m.eps) @ hw.T, -1)
            if lp_ref is None:
                lp_ref = lp
            r = res[name]
            r["kl"] += (lp_ref.exp() * (lp_ref - lp)).sum(-1).sum().item()
            r["top1"] += (lp.argmax(-1) == lp_ref.argmax(-1)).sum().item()
            r["nll"] += -lp.gather(-1, tgt[b][:, None]).sum().item()
        n += T - 1
    for r in res.values():
        r["kl"] /= n
        r["top1"] /= n
        r["ppl"] = math.exp(r.pop("nll") / n)
    out = {"model": args.model, "eval_tokens": n, "dark_dims_mid": dark.tolist(), "results": res}
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:50s} KL {v['kl']:.5f} top1 {v['top1']:.4f} ppl {v['ppl']:.3f}")


GG = {"attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o", "ffn_gate": "gate", "ffn_up": "up",
      "ffn_down": "down"}


class GGUFWeights:
    """dequantized linear weights of a GGUF (+ OA sidecar): W_eff = main with side columns put back, plus bias"""

    def __init__(self, path):
        import numpy as np
        from gguf import GGUFReader
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import oaquant
        self.np, self.oa = np, oaquant
        self.t = {t.name: t for t in GGUFReader(path).tensors}
        sc = path + ".oa.gguf"
        self.s = {t.name: t for t in GGUFReader(sc).tensors} if os.path.exists(sc) else {}

    def layer(self, i, base):
        np = self.np
        w = dict(base)
        for g, n in GG.items():
            name = f"blk.{i}.{g}.weight"
            t = self.t[name]
            N, K = int(t.shape[1]), int(t.shape[0])
            W = self.oa.dequantize(int(t.tensor_type), bytes(np.asarray(t.data).tobytes()), N, K)
            if name + ".oa_idx" in self.s:
                idx = np.asarray(self.s[name + ".oa_idx"].data).astype(np.int64)
                side = np.asarray(self.s[name + ".oa_side"].data).astype(np.float32).reshape(N, -1)
                W[:, idx] = side
                w["b_" + n] = torch.from_numpy(np.asarray(self.s[name + ".oa_bias"].data).astype(np.float32).copy())
            w[n] = torch.from_numpy(W)
        return w


def run_evalgguf(m, args):
    ids = chunks(m.tok, args.eval, args.T, skip=args.calib)
    cs = rope(args.T, m.hd, m.theta, m.cfg.get('rope_scaling'))
    emb = m.W.get("model.embed_tokens.weight")
    x0 = emb[ids]
    del emb
    names = ["ref (bf16)"] + [g.split("=")[0] for g in args.gguf]
    srcs = [None] + [GGUFWeights(g.split("=")[1]) for g in args.gguf]
    xs = [x0.clone() for _ in names]
    ident = lambda h, s: h
    for li in range(m.L):
        t0 = time.time()
        w = m.layer(li)
        for ci, s in enumerate(srcs):
            xs[ci] = m.block(xs[ci], w if s is None else s.layer(li, w), cs, ident, None, None, li)
        print(f"layer {li} {time.time() - t0:.1f}s", flush=True)
    nw, hw = m.head()
    res = {n: {"kl": 0.0, "top1": 0.0, "nll": 0.0} for n in names}
    B, T = ids.shape
    n = 0
    for b in range(B):
        lp_ref = None
        for ci, name in enumerate(names):
            lp = F.log_softmax(rms(xs[ci][b, :-1], nw, m.eps) @ hw.T, -1)
            if lp_ref is None:
                lp_ref = lp
            r = res[name]
            r["kl"] += (lp_ref.exp() * (lp_ref - lp)).sum(-1).sum().item()
            r["top1"] += (lp.argmax(-1) == lp_ref.argmax(-1)).sum().item()
            r["nll"] += -lp.gather(-1, ids[b, 1:][:, None]).sum().item()
        n += T - 1
    for r in res.values():
        r["kl"] /= n
        r["top1"] /= n
        r["ppl"] = math.exp(r.pop("nll") / n)
    json.dump({"model": args.model, "eval_tokens": n, "results": res}, open(args.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:40s} KL {v['kl']:.5f} top1 {v['top1']:.4f} ppl {v['ppl']:.3f}")


# ---------------------------------------------------------------- KV: channel statistics and k4c variants
def k4c_quant(k, bits):
    """k4c-style K: per channel, 32-token groups, asymmetric (min + scale) with `bits` per value; bits may be a
    per-(head, channel) tensor [KV, hd]. k: [B, T, KV, hd]"""
    B, T, H, D = k.shape
    g = k.reshape(B, T // 32, 32, H, D)
    lo, hi = g.amin(2, keepdim=True), g.amax(2, keepdim=True)
    lv = (2.0 ** (bits if torch.is_tensor(bits) else torch.tensor(float(bits)))) - 1
    s = ((hi - lo) / lv).clamp_min(1e-12)
    q = ((g - lo) / s).round().clamp(min=0) 
    q = torch.minimum(q, lv.expand_as(q) if torch.is_tensor(lv) and lv.dim() else lv)
    return (q * s + lo).reshape(B, T, H, D)


def q4_0_tokens(v, rot=0, keep=None, bits=4):
    """ggml Q4_0 (or Q8_0 with bits=8) along head_dim in blocks of 32, optional Hadamard of `rot` within each head
    first, optional channels kept exactly (keep: [KV, n] indices)"""
    x = v
    if keep is not None:
        kept = torch.gather(x, 3, keep[None, None].expand(x.shape[0], x.shape[1], -1, -1))
        x = x.scatter(3, keep[None, None].expand(x.shape[0], x.shape[1], -1, -1), 0.0)
    if rot:
        x = hadamard_blk(x, rot)
    sh = x.shape
    b = x.reshape(*sh[:-1], sh[-1] // 32, 32)
    if bits == 4:
        idx = b.abs().argmax(-1, keepdim=True)
        mx = torch.gather(b, -1, idx)
        d = mx / -8
        q = torch.where(d == 0, torch.zeros_like(b), (b / d + 8.5).floor().clamp(0, 15) - 8)
        y = (q * d).reshape(sh)
    else:
        d = b.abs().amax(-1, keepdim=True) / 127
        y = (torch.where(d == 0, torch.zeros_like(b), (b / d).round()) * d).reshape(sh)
    if rot:
        y = hadamard_blk(y, rot)
    if keep is not None:
        y = y.scatter(3, keep[None, None].expand(y.shape[0], y.shape[1], -1, -1), kept)
    return y


def hadamard_blk(x, b):
    key = ("blk", b)
    if key not in _HAD:
        H = torch.ones(1, 1)
        while H.shape[0] < b:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD[key] = H / math.sqrt(b)
    sh = x.shape
    return (x.reshape(*sh[:-1], sh[-1] // b, b) @ _HAD[key]).reshape(sh)


def alloc_bits(var, avg, lo=1, hi=8):
    """integer bits per channel minimizing sum var * 4^-b at mean `avg` (greedy: each extra bit to the channel whose
    error drops most)"""
    n = var.numel()
    b = torch.full((n,), float(lo))
    budget = int(round(avg * n)) - lo * n
    v = var.flatten().double()
    for _ in range(budget):
        gain = torch.where(b < hi, v * 4.0 ** (-b) * 0.75, torch.tensor(-1.0, dtype=torch.float64))
        b[gain.argmax()] += 1
    return b.reshape(var.shape)


def run_kv(m, args):
    cs = rope(args.T, m.hd, m.theta, m.cfg.get("rope_scaling"))
    emb = m.W.get("model.embed_tokens.weight")
    ident = lambda h, s: h
    # pass 1: K / V channel statistics on calibration text
    global TEXT
    t0txt = TEXT
    TEXT = args.calib_text or TEXT
    cal = chunks(m.tok, args.calib, args.T)
    TEXT = t0txt
    x = emb[cal]
    stats = []

    def grab(k, v, li):
        f = lambda t: t[:, 1:].reshape(-1, m.KV, m.hd)
        K, V = f(k), f(v)
        st = {}
        for nm, t in (("k", K), ("v", V)):
            mu, sd = t.mean(0), t.std(0)
            ms = t.pow(2).mean(0)
            # within 32-token groups: the range k4c's per-group min/scale has to cover
            g = t[: t.shape[0] // 32 * 32].reshape(-1, 32, m.KV, m.hd)
            gvar = g.var(1).mean(0)
            st[nm] = dict(mean=mu, std=sd, ms=ms, gvar=gvar,
                          first_absmax=(k if nm == "k" else v)[:, 0].abs().amax().item())
        stats.append(st)
        return k, v

    m.kvq = grab
    for li in range(m.L):
        x = m.block(x, m.layer(li), cs, ident, None, None, li)
    m.kvq = None
    summary = []
    for li, st in enumerate(stats):
        row = {"layer": li}
        for nm in ("k", "v"):
            s = st[nm]
            e = s["ms"].flatten()
            top = torch.sort(e, descending=True).values
            row[nm] = dict(
                top4_energy_share=(top[:4].sum() / e.sum()).item(),
                top8_energy_share=(top[:8].sum() / e.sum()).item(),
                max_over_median_rms=(top[0].sqrt() / e.median().sqrt()).item(),
                mean_energy_share=(s["mean"].pow(2).sum() / e.sum()).item(),  # energy in per-channel constants
                frac_channels_cv_below_0p3=((s["std"] / s["mean"].abs().clamp_min(1e-9)) < 0.3).float().mean().item(),
                # within-32-token-group variance vs total: what is left for k4c after its per-group min
                group_var_share=(s["gvar"].sum() / e.sum()).item(),
                first_tok_absmax=s["first_absmax"],
            )
        summary.append(row)
    # pass 2: KL of KV variants on test text
    ids = chunks(m.tok, args.eval, args.T, skip=0 if args.calib_text else args.calib)
    x0 = emb[ids]
    gv = [st["k"]["gvar"] for st in stats]
    vms = [st["v"]["ms"] for st in stats]
    bits_cache = {}

    def kbits(li, avg):
        if (li, avg) not in bits_cache:
            bits_cache[(li, avg)] = alloc_bits(gv[li], avg)
        return bits_cache[(li, avg)]

    def vkeep(li, n):
        return torch.argsort(vms[li], dim=1, descending=True)[:, :n]

    cfgs = {
        "ref": None,
        "k4c_q4 (K 4b/ch g32, V Q4_0)": lambda k, v, li: (k4c_quant(k, 4), q4_0_tokens(v)),
        "k4c_q4 + V Hadamard-64": lambda k, v, li: (k4c_quant(k, 4), q4_0_tokens(v, rot=64)),
        "k4c_q8 (V Q8_0)": lambda k, v, li: (k4c_quant(k, 4), q4_0_tokens(v, bits=8)),
        "K var-adaptive bits (avg 4) + V Q4_0 rot": lambda k, v, li: (k4c_quant(k, kbits(li, 4)), q4_0_tokens(v, rot=64)),
        "K 3b/ch + V Q4_0 rot": lambda k, v, li: (k4c_quant(k, 3), q4_0_tokens(v, rot=64)),
        "K var-adaptive bits (avg 3) + V Q4_0 rot": lambda k, v, li: (k4c_quant(k, kbits(li, 3)), q4_0_tokens(v, rot=64)),
        "K 2b/ch + V Q4_0 rot": lambda k, v, li: (k4c_quant(k, 2), q4_0_tokens(v, rot=64)),
        "K var-adaptive bits (avg 2) + V Q4_0 rot": lambda k, v, li: (k4c_quant(k, kbits(li, 2)), q4_0_tokens(v, rot=64)),
        "k4c_q4, V top-4 channels exact": lambda k, v, li: (k4c_quant(k, 4), q4_0_tokens(v, keep=vkeep(li, 4))),
    }
    names = list(cfgs)
    xs = [x0.clone() for _ in names]
    for li in range(m.L):
        w = m.layer(li)
        for ci, n in enumerate(names):
            m.kvq = cfgs[n]
            xs[ci] = m.block(xs[ci], w, cs, ident, None, None, li)
        m.kvq = None
        print(f"layer {li}", flush=True)
    nw, hw = m.head()
    res = {n: {"kl": 0.0, "top1": 0.0, "nll": 0.0} for n in names}
    B, T = ids.shape
    nt = 0
    for b in range(B):
        lp_ref = None
        for ci, n in enumerate(names):
            lp = F.log_softmax(rms(xs[ci][b, :-1], nw, m.eps) @ hw.T, -1)
            if lp_ref is None:
                lp_ref = lp
            r = res[n]
            r["kl"] += (lp_ref.exp() * (lp_ref - lp)).sum(-1).sum().item()
            r["top1"] += (lp.argmax(-1) == lp_ref.argmax(-1)).sum().item()
            r["nll"] += -lp.gather(-1, ids[b, 1:][:, None]).sum().item()
        nt += T - 1
    for r in res.values():
        r["kl"] /= nt
        r["top1"] /= nt
        r["ppl"] = math.exp(r.pop("nll") / nt)
    json.dump({"model": args.model, "stats": summary, "results": res}, open(args.out, "w"), indent=1)
    for li in (0, m.L // 4, m.L // 2, 3 * m.L // 4, m.L - 1):
        print(summary[li])
    for k, v in res.items():
        print(f"{k:45s} KL {v['kl']:.5f} top1 {v['top1']:.4f} ppl {v['ppl']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["stats", "kl", "check", "evalgguf", "kv"], required=True)
    ap.add_argument("--out")
    ap.add_argument("--stats")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--calib", type=int, default=4)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--calib-text", default="")
    ap.add_argument("--gguf", nargs="*", default=[], help="evalgguf: NAME=FILE[.gguf] (sidecar FILE.oa.gguf used if present)")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    if args.text:
        TEXT = args.text
    m = Model(args.model)
    if args.mode == "stats":
        run_stats(m, args)
    elif args.mode == "evalgguf":
        run_evalgguf(m, args)
    elif args.mode == "kv":
        run_kv(m, args)
    elif args.mode == "kl":
        run_kl(m, args)
    else:
        from transformers import AutoModelForCausalLM
        ids = chunks(m.tok, 1, 128)
        hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        ref = hf(ids).logits
        emb = m.W.get("model.embed_tokens.weight")
        x = emb[ids]
        cs = rope(128, m.hd, m.theta, m.cfg.get('rope_scaling'))
        for li in range(m.L):
            x = m.block(x, m.layer(li), cs, lambda h, s: h, None)
        nw, hw = m.head()
        mine = rms(x, nw, m.eps) @ hw.T
        print("max |logit diff|", (mine - ref).abs().max().item(), "ref absmax", ref.abs().max().item())
