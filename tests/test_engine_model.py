"""kurn model engine v2 end to end: tiny synthetic Qwen3 (dense) and OLMoE (MoE) GGUFs with random
Q8_0 (or all-Q4_0) weights are compiled into engines and run at thread counts covering all three KV-head
assignment modes (whole heads per thread, threads sharing a KV head, uneven / idle threads).
Every step's logits are compared with an independent numpy forward pass that follows ggml's
numerics (Q8_0 activations, f16 KV cache, NEOX RoPE)."""

import atexit
import os
import shutil
import subprocess
import tempfile
from unittest import mock

import numpy as np
import pytest

from kurn import toolchain

gguf = pytest.importorskip("gguf")

pytestmark = pytest.mark.skipif("avx512_vnni" not in toolchain.cpu_flags(), reason="host CPU lacks AVX-512 VNNI")

DIMS = dict(n_layer=2, n_embd=256, n_head=4, n_kv=2, head_dim=64, n_ff=512, vocab=1000, eps=1e-6, base=10000.0)
MOE = dict(n_expert=8, n_used=2, n_ff=256)
PROMPT = [3, 141, 59, 265, 358, 979]
NGEN = 6
_OUT = tempfile.mkdtemp(prefix="kurn-engine-model-")  # per process (pytest-xdist workers)
atexit.register(shutil.rmtree, _OUT, True)


# ---------------------------------------------------------------- Q8_0 helpers (ggml semantics)
def q8_quantize(x):
    """ggml x86 quantize_row_q8_0: d = amax/127, q = round-half-even(x / d) with 1/d precomputed."""
    x = np.asarray(x, np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = (amax / np.float32(127)).astype(np.float32)
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.rint(x * inv[:, None]).astype(np.int8)
    return d.astype(np.float16), q


def q8_dot(wd, wq, x):
    """W (row blocks: f16 scales [R, B], int8 [R, B, 32]) times Q8_0-quantized x."""
    xd, xq = q8_quantize(x)
    s = np.einsum("rbk,bk->rb", wq.astype(np.int64), xq.astype(np.int64)).astype(np.float64)
    return (s * wd.astype(np.float64) * xd.astype(np.float64)[None, :]).sum(axis=1).astype(np.float32)


def q_weight(rng, rows, cols, scale, fmt):
    """Random weight quantized by gguf; returns the raw bytes and (f16 scales, integer values) per block."""
    w = (rng.standard_normal((rows, cols)) * scale).astype(np.float32)
    if fmt == "q8_0":
        raw = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q8_0)
        blk = raw.reshape(rows, cols // 32, 34)
        return raw, (blk[..., :2].copy().view(np.float16)[..., 0], blk[..., 2:].copy().view(np.int8))
    raw = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q4_0)
    blk = raw.reshape(rows, cols // 32, 18)
    qs = blk[..., 2:].astype(np.int8)
    return raw, (blk[..., :2].copy().view(np.float16)[..., 0], np.concatenate([(qs & 15) - 8, (qs >> 4 & 15) - 8], axis=-1))


def rms_scale(x, eps):
    ss = np.sum((x.astype(np.float32) * x.astype(np.float32)).astype(np.float64))
    return np.float32(1) / np.sqrt(np.float32(ss / x.size) + np.float32(eps))


def rmsnorm(x, w, eps):
    return ((x * rms_scale(x, eps)).astype(np.float32) * w).astype(np.float32)


def rope_neox(v, pos, D, base):
    th = pos * np.power(np.float64(base), -2.0 * np.arange(D // 2) / D)
    c, s = np.cos(th), np.sin(th)
    a, b = v[..., : D // 2].astype(np.float64), v[..., D // 2:].astype(np.float64)
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1).astype(np.float32)


# ---------------------------------------------------------------- model
def make_model(path, arch, seed, fmt="q8_0"):
    rng = np.random.default_rng(seed)
    p = DIMS | (MOE if arch == "olmoe" else {})
    L, E, H, KV, D, F, V = p["n_layer"], p["n_embd"], p["n_head"], p["n_kv"], p["head_dim"], p["n_ff"], p["vocab"]
    w = gguf.GGUFWriter(path, arch)
    w.add_uint32(f"{arch}.block_count", L)
    w.add_uint32(f"{arch}.embedding_length", E)
    w.add_uint32(f"{arch}.attention.head_count", H)
    w.add_uint32(f"{arch}.attention.head_count_kv", KV)
    w.add_uint32(f"{arch}.attention.key_length", D)
    w.add_uint32(f"{arch}.feed_forward_length", F)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", p["eps"])
    w.add_float32(f"{arch}.rope.freq_base", p["base"])
    if arch == "olmoe":
        w.add_uint32(f"{arch}.expert_count", p["n_expert"])
        w.add_uint32(f"{arch}.expert_used_count", p["n_used"])
    m = {"p": p, "arch": arch, "layers": []}
    qtype = gguf.GGMLQuantizationType.Q8_0 if fmt == "q8_0" else gguf.GGMLQuantizationType.Q4_0

    def q8(name, rows, cols, scale):
        raw, wb = q_weight(rng, rows, cols, scale, fmt)
        w.add_tensor(name, raw, raw_dtype=qtype)
        return wb

    def f32(name, n, around=1.0, spread=0.2):
        v = (around + spread * rng.standard_normal(n)).astype(np.float32)
        w.add_tensor(name, v)
        return v

    m["embd"] = q8("token_embd.weight", V, E, 0.25)
    m["output"] = q8("output.weight", V, E, E ** -0.5) if arch == "olmoe" else m["embd"]  # Qwen3: tied
    m["out_norm"] = f32("output_norm.weight", E)
    for i in range(L):
        b, ly = f"blk.{i}.", {}
        ly["attn_norm"] = f32(b + "attn_norm.weight", E)
        ly["q"] = q8(b + "attn_q.weight", H * D, E, E ** -0.5)
        ly["k"] = q8(b + "attn_k.weight", KV * D, E, E ** -0.5)
        ly["v"] = q8(b + "attn_v.weight", KV * D, E, E ** -0.5)
        ly["o"] = q8(b + "attn_output.weight", E, H * D, (H * D) ** -0.5)
        nq, nk = (D, D) if arch == "qwen3" else (H * D, KV * D)
        ly["q_norm"] = f32(b + "attn_q_norm.weight", nq)
        ly["k_norm"] = f32(b + "attn_k_norm.weight", nk)
        ly["ffn_norm"] = f32(b + "ffn_norm.weight", E)
        if arch == "qwen3":
            ly["gate"] = q8(b + "ffn_gate.weight", F, E, E ** -0.5)
            ly["up"] = q8(b + "ffn_up.weight", F, E, E ** -0.5)
            ly["down"] = q8(b + "ffn_down.weight", E, F, F ** -0.5)
        else:
            X = p["n_expert"]
            ly["router"] = f32(b + "ffn_gate_inp.weight", X * E, 0.0, 0.3).reshape(X, E)
            for nm, rows, cols in (("gate", F, E), ("up", F, E), ("down", E, F)):
                raws, wbs = zip(*(q_weight(rng, rows, cols, cols ** -0.5, fmt) for _ in range(X)))
                w.add_tensor(b + f"ffn_{nm}_exps.weight", np.stack(raws), raw_dtype=qtype)
                ly[nm] = wbs
        m["layers"].append(ly)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return m


def swiglu_ffn(gate, up, down, xn):
    g, u = q8_dot(*gate, xn), q8_dot(*up, xn)
    return q8_dot(*down, (g / (1 + np.exp(-g.astype(np.float64)))).astype(np.float32) * u)


def reference(m, tokens):
    """Teacher-forced logits after each token."""
    p, arch = m["p"], m["arch"]
    E, H, KV, D, eps = p["n_embd"], p["n_head"], p["n_kv"], p["head_dim"], p["eps"]
    kc = [[] for _ in m["layers"]]
    vc = [[] for _ in m["layers"]]
    out = []
    for pos, tok in enumerate(tokens):
        wd, wq = m["embd"]
        h = (wd[tok].astype(np.float32)[:, None] * wq[tok].astype(np.float32)).reshape(-1)
        for i, ly in enumerate(m["layers"]):
            xn = rmsnorm(h, ly["attn_norm"], eps)
            q, k, v = q8_dot(*ly["q"], xn), q8_dot(*ly["k"], xn), q8_dot(*ly["v"], xn)
            if arch == "qwen3":
                q = np.stack([rmsnorm(r, ly["q_norm"], eps) for r in q.reshape(H, D)])
                k = np.stack([rmsnorm(r, ly["k_norm"], eps) for r in k.reshape(KV, D)])
            else:
                q, k = rmsnorm(q, ly["q_norm"], eps).reshape(H, D), rmsnorm(k, ly["k_norm"], eps).reshape(KV, D)
            q, k = rope_neox(q, pos, D, p["base"]), rope_neox(k, pos, D, p["base"])
            kc[i].append(k.astype(np.float16).astype(np.float64))
            vc[i].append(v.reshape(KV, D).astype(np.float16).astype(np.float64))
            K, Vv = np.stack(kc[i], 1), np.stack(vc[i], 1)  # [KV, pos+1, D]
            att = np.empty((H, D), np.float32)
            for hh in range(H):
                s = K[hh // (H // KV)] @ q[hh].astype(np.float64) / np.sqrt(D)
                e = np.exp(s - s.max())
                att[hh] = (e / e.sum()) @ Vv[hh // (H // KV)]
            h = h + q8_dot(*ly["o"], att.reshape(-1))
            xn = rmsnorm(h, ly["ffn_norm"], eps)
            if arch == "qwen3":
                h = h + swiglu_ffn(ly["gate"], ly["up"], ly["down"], xn)
            else:
                lg = ly["router"].astype(np.float64) @ xn.astype(np.float64)
                pr = np.exp(lg - lg.max())
                pr /= pr.sum()
                y = np.zeros(E, np.float64)
                for x in np.argsort(-pr, kind="stable")[: p["n_used"]]:
                    y += pr[x] * swiglu_ffn(ly["gate"][x], ly["up"][x], ly["down"][x], xn)
                h = h + y.astype(np.float32)
        out.append(q8_dot(*m["output"], rmsnorm(h, m["out_norm"], eps)))
    return np.stack(out)


# ---------------------------------------------------------------- engine runs
_MODELS = {}


def _model(arch):
    """arch: qwen3 | olmoe, optionally suffixed -q4 for an all-Q4_0 model."""
    if arch not in _MODELS:
        base, _, q4 = arch.partition("-")
        path = os.path.join(_OUT, f"tiny-{arch}-{os.getpid()}.gguf")
        _MODELS[arch] = (path, make_model(path, base, seed=7 if base == "qwen3" else 11, fmt="q4_0" if q4 else "q8_0"))
    return _MODELS[arch]


def _engine(arch, defines=()):
    from kurn.model.compile_model import compile_model

    path, _ = _model(arch)
    tag = "-".join(d.replace("=", "") for d in defines) or "default"
    with mock.patch.dict(os.environ, {"KURN_CACHE_DIR": _OUT}):
        exe, _, _ = compile_model(path, out=os.path.join(_OUT, f"engine-{arch}-{tag}"), defines=defines)
    return exe


def _run(exe, arch, T, extra_env=None):
    path, m = _model(arch)
    dump = f"{exe}.T{T}.logits"
    env = {**os.environ, "KURN_WAIT": "futex:2000", "KURN_THP": "0", "KURN_DUMP_LOGITS": dump, **(extra_env or {})}
    r = subprocess.run([exe, path, "gen", str(T), str(NGEN), ",".join(map(str, PROMPT))], env=env,
                       capture_output=True, text=True, timeout=300, check=True)
    gen = [int(t) for t in r.stdout.split("gen:")[1].split("\n")[0].split()]
    logits = np.fromfile(dump, np.float32).reshape(-1, m["p"]["vocab"])
    return gen, logits, r.stdout


def _check(arch, gen, logits):
    _, m = _model(arch)
    assert logits.shape[0] == len(PROMPT) + NGEN
    assert gen[0] == int(np.argmax(logits[len(PROMPT) - 1]))
    for i in range(1, NGEN):
        assert gen[i] == int(np.argmax(logits[len(PROMPT) + i - 1]))
    ref = reference(m, PROMPT + gen)  # the engine's own greedy continuation, teacher-forced
    err = np.abs(ref - logits[: len(ref)]).max(axis=1) / ref.std(axis=1)
    # float sums in a different order can flip a Q8_0 activation rounding at a .5 boundary:
    # that shifts one step's logits slightly; anything structural would be off everywhere
    assert err.max() < 0.1 and np.median(err) < 1e-4, err


@pytest.mark.parametrize("arch", ["qwen3", "olmoe"])
@pytest.mark.parametrize("T", [1, 2, 3, 4, 8])
def test_engine_matches_numpy_reference(arch, T):
    gen, logits, out = _run(_engine(arch), arch, T)
    _check(arch, gen, logits)
    syncs = float(out.split("barriers_per_tok ")[1].split()[0])
    per_layer = 3 if arch == "olmoe" else 2
    assert syncs == DIMS["n_layer"] * per_layer + 1


@pytest.mark.parametrize("arch", ["qwen3-q4", "olmoe-q4"])
@pytest.mark.parametrize("T", [1, 3, 8])
def test_engine_q4_0_weights_match_numpy_reference(arch, T):
    gen, logits, _ = _run(_engine(arch), arch, T)
    _check(arch, gen, logits)


def test_mixed_weight_formats_rejected():
    from kurn.model.compile_model import config_header

    path = os.path.join(_OUT, f"mixed-{os.getpid()}.gguf")
    w = gguf.GGUFWriter(path, "qwen3")
    for k in ("block_count", "embedding_length", "attention.head_count", "attention.head_count_kv", "feed_forward_length"):
        w.add_uint32(f"qwen3.{k}", 1)
    rng = np.random.default_rng(0)
    w.add_tensor("token_embd.weight", q_weight(rng, 16, 32, 1.0, "q4_0")[0], raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    w.add_tensor("blk.0.attn_q.weight", q_weight(rng, 16, 32, 1.0, "q8_0")[0], raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="all matmul weights Q8_0"):
        config_header(path)


@pytest.mark.parametrize("arch", ["qwen3", "olmoe", "olmoe-q4"])
def test_fused_epilogue_bit_identical_in_engine(arch):
    _, fused, _ = _run(_engine(arch), arch, 4)
    _, unfused, _ = _run(_engine(arch, ("KURN_EPILOGUE=0",)), arch, 4)
    assert np.array_equal(fused, unfused)


def test_wait_policies_and_prefetch_in_wait_do_not_change_results():
    exe = _engine("qwen3")
    _, a, _ = _run(exe, "qwen3", 4, {"KURN_WAIT": "attn=yield:50,ffn=futex:0,qk=futex,out=yield", "KURN_PFWAIT": "65536"})
    _, b, _ = _run(exe, "qwen3", 4)
    assert np.array_equal(a, b)
