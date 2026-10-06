"""kurn attention as llama.cpp's FLASH_ATTN_EXT (integration/llama.cpp/gen_ggml_attn.py, kurn-attn.cpp).

- the generator emits every variant with renamed symbols and a matching dispatch table;
- the renamed kernels pass kurn's own numerical check (bench_attn harness, awkward shapes);
- the `exact` variant is batch-invariant: a query row's output is bit-identical whether it is computed
  alone or inside a batch, for any KV length (padding hidden by the mask) and thread count;
- with a built llama.cpp checkout (KURN_LLAMA_CPP), ggml's test-backend-ops FLASH_ATTN_EXT cases pass
  against ggml's reference implementation with kurn taking the supported nodes."""

import ctypes
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kurn import attention as A

from conftest import ROOT, require_tree

require_tree("integration/llama.cpp/gen_ggml_attn.py")
INTEG = ROOT / "integration" / "llama.cpp"
sys.path.insert(0, str(INTEG))
import gen_ggml_attn as gen  # noqa: E402

np = pytest.importorskip("numpy")


def test_generator_writes_every_variant_with_renamed_symbols(tmp_path):
    gen.main([str(tmp_path)])
    disp = (tmp_path / "kattn_dispatch.h").read_text()
    assert (tmp_path / "kurn_attn.h").exists()
    n = 0
    for variant in gen.VARIANTS:
        for kv in gen.KV:
            for dk, dv in gen.DIMS:
                name = gen.name(variant, kv, dk, dv)
                src = (tmp_path / f"{name}.c").read_text()
                assert src.startswith("#if ") and src.rstrip().endswith("#endif")
                for s in gen.SYMBOLS:
                    assert f"#define {s} {name}{s[len('kattn'):]}\n" in src
                assert f'"{name}"' in disp
                n += 1
    assert disp.count("{ \"kattn_") == n
    # exact = f32 tile engine for every n_q (no decode row engine) and a single KV split
    src = (tmp_path / "kattn_exact_f16_d128.c").read_text()
    assert "#define KA_DEC_ROWS 0" in src and "#define KA_SPLIT 1" in src and "#define KA_ENGINE 0" in src


def test_exact_cannot_be_overridden_out_of_exactness():
    c = gen.config("exact", "f16", 128, 128, {"dec_rows": 8, "split": 0, "tile_kv": 64})
    assert (c["dec_rows"], c["split"], c["tile_kv"]) == (0, 1, 64)


SHIM = """
#undef kattn
#undef kattn_workspace
#undef kattn_config
#undef kattn_pack_bytes
#undef kattn_pack
#undef kattn_packed
size_t kattn_workspace(const kattn_args *a, int nth) { return NAME_workspace(a, nth); }
void kattn(const kattn_args *a, void *ws, int ith, int nth) { NAME(a, ws, ith, nth); }
int64_t kattn_config(int *dk, int *dv, int *kv) { return NAME_config(dk, dv, kv); }
size_t kattn_pack_bytes(const kattn_args *a, int64_t cap) { return NAME_pack_bytes(a, cap); }
void kattn_pack(const kattn_args *a, void *p, int64_t cap, int64_t j0, int64_t j1, int ith, int nth) { NAME_pack(a, p, cap, j0, j1, ith, nth); }
void kattn_packed(const kattn_args *a, const void *p, int64_t cap, void *ws, int ith, int nth) { NAME_packed(a, p, cap, ws, ith, nth); }
"""


def _build(variant, kv="f16", dk=128, dv=128):
    """The generated (renamed) kernel plus unrenamed forwarders, as a kurn_attn.h library."""
    c, name, src = gen.source(variant, kv, dk, dv)
    ok, missing = A.runnable(c["target"])
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    so = A._compile(src + SHIM.replace("NAME", name), A.TARGETS[c["target"]][0], f"llama_{name}")
    return c, so


@pytest.mark.parametrize("variant,kv", [("f32", "f16"), ("f32", "q8_0"), ("exact", "f16"), ("exact", "q8_0"), ("exact", "bf16"),
                                        ("bf16", "f16"), ("amx", "q8_0")])  # fmt: skip
def test_generated_kernels_pass_kurn_check(variant, kv):
    c, so = _build(variant, kv)
    worst = A.check(so, c)
    assert worst["check"] != "FAIL", worst


class Args(ctypes.Structure):
    _fields_ = [
        ("n_q", ctypes.c_int64), ("n_kv", ctypes.c_int64), ("q_pos0", ctypes.c_int64),
        ("n_head", ctypes.c_int32), ("n_head_kv", ctypes.c_int32), ("causal", ctypes.c_int32), ("scale", ctypes.c_float),
        ("q", ctypes.c_void_p), ("q_s_tok", ctypes.c_int64), ("q_s_head", ctypes.c_int64),
        ("k", ctypes.c_void_p), ("k_s_tok", ctypes.c_int64), ("k_s_head", ctypes.c_int64),
        ("v", ctypes.c_void_p), ("v_s_tok", ctypes.c_int64), ("v_s_head", ctypes.c_int64),
        ("mask", ctypes.c_void_p), ("mask_s_tok", ctypes.c_int64),
        ("out", ctypes.c_void_p), ("o_s_tok", ctypes.c_int64), ("o_s_head", ctypes.c_int64),
        ("k_tail", ctypes.c_void_p), ("kt_s_tok", ctypes.c_int64), ("kt_s_head", ctypes.c_int64),
        ("rope_freq", ctypes.c_void_p), ("k_pos0", ctypes.c_int64), ("rope_dim", ctypes.c_int32), ("rope_mode", ctypes.c_int32),
    ]  # fmt: skip


def _run(lib, q, k, v, mask, n_kv, nth):
    """kattn on q [n_q][nh][d] f32 against the first n_kv rows of k, v [cells][nhkv][d] f16, with an
    optional ggml-style f16 mask [n_q][>= n_kv]; causal = 0 as in kurn-attn.cpp."""
    out = np.zeros((q.shape[0], q.shape[1], v.shape[2]), np.float32)
    a = Args(q.shape[0], n_kv, 0, q.shape[1], k.shape[1], 0, 1 / np.sqrt(q.shape[2]), q.ctypes.data, q.strides[0] // 4,
             q.strides[1] // 4, k.ctypes.data, k.strides[0], k.strides[1], v.ctypes.data, v.strides[0], v.strides[1],
             mask.ctypes.data if mask is not None else None, mask.strides[0] // 2 if mask is not None else 0,
             out.ctypes.data, out.strides[0] // 4, out.strides[1] // 4)  # fmt: skip
    ws = np.zeros(lib.kattn_workspace(ctypes.byref(a), nth) + 64, np.uint8)
    ts = [threading.Thread(target=lib.kattn, args=(ctypes.byref(a), ws.ctypes.data, i, nth)) for i in range(nth)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def _causal_mask(n_q, width, pos0):
    m = np.zeros((n_q, width), np.float16)
    for t in range(n_q):
        m[t, pos0 + t + 1 :] = -np.inf
    return m


def test_exact_kernel_is_batch_invariant():
    """What speculative decoding needs: row t of a verify batch == the same token decoded alone."""
    c, so = _build("exact")
    lib = ctypes.CDLL(so)
    lib.kattn_workspace.argtypes, lib.kattn_workspace.restype = [ctypes.POINTER(Args), ctypes.c_int], ctypes.c_size_t
    lib.kattn.argtypes = [ctypes.POINTER(Args), ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    rng = np.random.default_rng(7)
    cells, nh, nhkv, d, pos0, nb = 1280, 16, 8, 128, 517, 9
    k = rng.standard_normal((cells, nhkv, d)).astype(np.float16)
    v = rng.standard_normal((cells, nhkv, d)).astype(np.float16)
    q = (3 * rng.standard_normal((nb, nh, d))).astype(np.float32)
    batch = _run(lib, q, k, v, _causal_mask(nb, cells, pos0), 1024, 3)  # llama.cpp-style padded n_kv
    for t in range(nb):
        for n_kv, nth in ((pos0 + t + 1, 1), (768, 4), (1280, 8)):
            one = _run(lib, np.ascontiguousarray(q[t : t + 1]), k, v, _causal_mask(1, cells, pos0 + t), n_kv, nth)
            assert np.array_equal(one[0], batch[t]), (t, n_kv, nth, np.abs(one[0] - batch[t]).max())


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    exe = d / "build" / "bin" / "test-backend-ops"
    if not (d / "ggml" / "src" / "ggml-cpu" / "kurn" / "kurn-attn.cpp").exists() or not exe.exists():
        return None
    return d


@pytest.mark.skipif(_llama_dir() is None, reason="no built llama.cpp checkout with kurn attention (KURN_LLAMA_CPP)")
@pytest.mark.parametrize("mode", [{}, {"GGML_KURN_FA_MODE": "exact"}, {"GGML_KURN_FA_ENGINE": "bf16"}])
def test_flash_attn_ext_matches_ggml_reference(mode):
    """test-backend-ops compares the CPU backend against itself in reference mode (use_ref), where
    kurn steps aside: so this is kurn's FLASH_ATTN_EXT against ggml's own vec kernel (NMSE < 5e-4)."""
    exe = _llama_dir() / "build" / "bin" / "test-backend-ops"
    env = dict(os.environ, GGML_KURN_VERBOSE="1", GGML_KURN_AMX="0", **mode)
    flt = r"hsk=(64|128),hsv=(64|128),nh=4,nr23=\[(1|4),(1|3)\],kv=(113|512),nb=(1|3|75)"
    r = subprocess.run([str(exe), "-o", "FLASH_ATTN_EXT", "-b", "CPU", "-p", flt], capture_output=True, text=True, timeout=1200,
                       env=env)  # fmt: skip
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    m = re.search(r"kurn fa: (\d+) FLASH_ATTN_EXT nodes on kurn, (\d+) on ggml", r.stderr)
    assert m and int(m.group(1)) >= 20, r.stderr[-2000:]
    assert "FAIL" not in r.stdout
