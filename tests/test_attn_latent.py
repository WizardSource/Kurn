"""kurn.latent: absorbed MLA on the kurn attn kernel equals ordinary multi-head attention over the
decompressed per-head K/V (float64), on random weights with the DeepSeek / PLM shapes."""

import pytest

from kurn import attention as A
from kurn import latent

np = pytest.importorskip("numpy")

np = pytest.importorskip("numpy")


def _mha(q_nope, q_pe, c, k_pe, w_uk, w_uv, scale):
    k_nope = np.einsum("hnc,jc->jhn", w_uk, c)
    v = np.einsum("hvc,jc->jhv", w_uv, c)
    T, H, _ = q_nope.shape
    N = c.shape[0]
    out = np.zeros((T, H, w_uv.shape[1]))
    for t in range(T):
        lim = N - T + t + 1
        for h in range(H):
            s = (k_nope[:lim, h] @ q_nope[t, h] + k_pe[:lim] @ q_pe[t, h]) * scale
            p = np.exp(s - s.max())
            out[t, h] = p @ v[:lim, h] / p.sum()
    return out


@pytest.mark.parametrize("nq,nkv,heads", [(1, 700, 16), (37, 300, 4)])
def test_absorbed_mla_matches_decompressed_attention(nq, nkv, heads):
    if not A.runnable("avx512")[0]:
        pytest.skip("needs AVX-512")
    rng = np.random.default_rng(5)
    dn, dr, dv, dc = 128, 64, 128, 512
    w_uk = rng.standard_normal((heads, dn, dc)) / np.sqrt(dc)
    w_uv = rng.standard_normal((heads, dv, dc)) / np.sqrt(dc)
    c = rng.standard_normal((nkv, dc)).astype(np.float16).astype(np.float64)
    k_pe = rng.standard_normal((nkv, dr)).astype(np.float16).astype(np.float64)
    q_nope = 2 * rng.standard_normal((nq, heads, dn))
    q_pe = 2 * rng.standard_normal((nq, heads, dr))
    scale = 1 / np.sqrt(dn + dr)
    ref = _mha(q_nope, q_pe, c, k_pe, w_uk, w_uv, scale)

    w = np.concatenate([w_uk, w_uv], axis=1).reshape(heads * (dn + dv), dc)
    uk, uv = latent.split_kv_b(w, heads, dn, dv)
    qt = latent.absorb_queries(q_nope, q_pe, uk)
    cache = latent.latent_cache(c, k_pe)
    lib = latent.load(A.build(A.resolve({"target": "avx512", "kv": "f16", "dk": 576, "mla": 1})))
    out = latent.up_project(latent.run_kattn(lib, qt, cache, None, dc, scale, threads=3), uv)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-4
    o64 = latent.reference(qt, cache, dc, scale)
    assert np.abs(latent.up_project(o64, uv) - ref).max() / np.abs(ref).max() < 1e-5


def test_kv_bytes():
    b = latent.kv_bytes_per_token(32, n_head_kv=8)
    assert b["mla_latent"] == 32 * 576 * 2
    assert b["mla_decompressed"] == 32 * 16 * 320 * 2
    assert b["gqa"] == 32 * 2 * 8 * 128 * 2
