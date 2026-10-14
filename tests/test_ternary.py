"""Ternary weights as presence bitmap + packed signs (kurn.ternary, integration/llama.cpp/ggml-kurn/kurn-tbm.cpp):
size model, reference encoder / decoder, native TQ1_0 / TQ2_0 decoding against gguf-py, the zero-fraction scan,
and (on AVX-512 VNNI + BMI2 hosts with g++) the C packer byte for byte against the reference encoder and its
kernels against the TQ2_0 reference."""

import ctypes
import json
import random
import shutil
import struct
import subprocess

import pytest

np = pytest.importorskip("numpy")

from kurn import hooks, kernels  # noqa: E402,F401  (kernels loads kurn.ext)
from kurn import ternary as T  # noqa: E402
from kurn.formats import reference_gemm  # noqa: E402
from kurn.targets import TARGETS  # noqa: E402
from kurn.toolchain import cpu_flags  # noqa: E402

from conftest import ROOT, q8_K_blocks  # noqa: E402

TBM_CPP = ROOT / "integration" / "llama.cpp" / "ggml-kurn" / "kurn-tbm.cpp"


def _trits(rng, n, k, z):
    t = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, k))
    t[rng.random((n, k)) < z] = 0
    return t


def _d(rng, n, nb):
    return (((9 + rng.integers(0, 5, size=(n, nb))) << 10) | rng.integers(0, 1024, size=(n, nb))).astype(np.uint16)


def _tq2_0_bytes(t, d):
    """trits [N, K] + fp16 bits [N, K / 256] -> ggml TQ2_0 rows."""
    n, k = t.shape
    out = bytearray()
    for r in range(n):
        for b in range(k // 256):
            q = (t[r, 256 * b : 256 * b + 256] + 1).astype(np.uint8)
            qs = bytearray(64)
            for v in range(256):
                qs[(v // 128) * 32 + v % 32] |= int(q[v]) << (2 * ((v % 128) // 32))
            out += qs + struct.pack("<H", int(d[r, b]))
    return bytes(out)


def test_size_model():
    assert T.TQ1_BPW == 1.6875 and T.TQ2_BPW == 2.0625
    assert abs(T.breakeven_z() - 0.375) < 1e-12
    assert T.tbm_bpw(0.0) == T.TQ2_BPW
    assert abs(T.tbm_bpw(0.515) - 1.5475) < 1e-12  # 2 - z + 1/16 (scale), the paper's 1.485 + 0.0625


@pytest.mark.parametrize("z", [0.0, 0.02, 0.42, 0.97])
@pytest.mark.parametrize("n,k", [(1, 256), (17, 512), (40, 256)])
def test_encode_decode_roundtrip(z, n, k):
    rng = np.random.default_rng(int(z * 100) + n + k)
    t, d = _trits(rng, n, k, z), _d(rng, n, k // 256)
    buf = T.encode(t, d)
    h = T.header(buf)
    assert (h["K"], h["N"], h["nrec_k"], h["ngroups"], h["used"]) == (k, n, k // 256, (n + 15) // 16, len(buf))
    t2, d2 = T.decode(buf)
    assert (t2 == t).all() and (d2 == d).all()
    nz = int(np.count_nonzero(t))
    recs = h["ngroups"] * h["nrec_k"]
    assert len(buf) <= T.HDR + (8 * (h["ngroups"] + 1) + 63) // 64 * 64 + recs * T.REC_FIXED + (nz + 7 * recs) // 8 + recs + T.SLACK
    if z == 0.0:
        assert h["dense"] == 0  # every word full: sign offsets stay byte-aligned, the 64-bit window always holds them
    if z == 0.02 and n >= 16:
        assert h["dense"] > 0  # near-full words at unaligned offsets exceed the one-load window (57..64 bits)
    if z > 0.9:
        assert h["dense"] == 0


@pytest.mark.parametrize("qtype", ["TQ1_0", "TQ2_0"])
def test_native_trits_match_gguf(qtype):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize, quantize

    rng = np.random.default_rng(3)
    t = _trits(rng, 8, 512, 0.4)
    t[:, ::256] = 1  # one +-1 per block: d = scale exactly
    f = t.astype(np.float32) * 0.03
    q = quantize(f, GGMLQuantizationType[qtype])
    raw = q.tobytes()
    nb = 8 * 2
    fn = T.tq1_0_trits if qtype == "TQ1_0" else T.tq2_0_trits
    trits, d = fn(raw, nb)
    assert (trits.reshape(8, 512) == t).all()
    deq = dequantize(q, GGMLQuantizationType[qtype]).reshape(8, 512)
    dd = d.view(np.float16).astype(np.float32).reshape(8, 2)
    assert np.allclose(deq, trits.reshape(8, 512) * np.repeat(dd, 256, axis=1))
    zc = (T.zero_count_tq1_0 if qtype == "TQ1_0" else T.zero_count_tq2_0)(raw, nb)
    assert zc == (int(np.count_nonzero(t == 0)), 8 * 512)


def test_scan_gguf(tmp_path):
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quantize

    rng = np.random.default_rng(5)
    p = tmp_path / "t.gguf"
    w = GGUFWriter(str(p), "llama")
    zs = {}
    for name, qt, z in (("a.weight", "TQ2_0", 0.45), ("b.weight", "TQ1_0", 0.2)):
        t = _trits(rng, 16, 512, z)
        t[:, ::256] = 1
        zs[name] = int(np.count_nonzero(t == 0))
        q = quantize(t.astype(np.float32) * 0.02, GGMLQuantizationType[qt])
        w.add_tensor(name, q, raw_dtype=GGMLQuantizationType[qt])
    w.add_tensor("norm.weight", np.ones(512, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    rows = {r["name"]: r for r in T.scan(str(p))}
    assert set(rows) == {"a.weight", "b.weight"}
    for name, zc in zs.items():
        assert rows[name]["zeros"] == zc and rows[name]["n"] == 16 * 512
    s = T.summarize(list(rows.values()))
    assert s["tensors_beating_tq1"] == 1
    assert "ternary" in hooks.COMMANDS
    assert hooks.COMMANDS["ternary"](["stats", str(p)]) == 0


def test_scan_packed_safetensors(tmp_path):
    codes = np.array([0, 1, 2, 1, 1, 0, 2, 2], dtype=np.uint8)  # four codes per byte, value = code - 1
    packed = (codes[0::4] | codes[1::4] << 2 | codes[2::4] << 4 | codes[3::4] << 6).astype(np.uint8)
    hdr = json.dumps({"w": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(hdr)) + hdr + packed.tobytes())
    (r,) = T.scan(str(p))
    assert r["n"] == 8 and r["zeros"] == 3


HAVE_ISA = TARGETS["avx512_vnni"].requires <= cpu_flags() and "bmi2" in cpu_flags()


@pytest.fixture(scope="module")
def tbm_lib(tmp_path_factory):
    if not (HAVE_ISA and shutil.which("g++") and TBM_CPP.exists()):
        pytest.skip("needs g++, kurn-tbm.cpp and an AVX-512 VNNI + BMI2 CPU")
    d = tmp_path_factory.mktemp("tbm")
    lib = d / "libtbm.so"
    subprocess.run(["g++", "-O2", "-march=native", "-mno-avx512fp16", "-std=c++17", "-shared", "-fPIC", str(TBM_CPP), "-o", str(lib)],
                   check=True)  # fmt: skip
    so = ctypes.CDLL(str(lib))
    i64, vp, sz = ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t
    so.kurn_tbm_bytes.restype = sz
    so.kurn_tbm_bytes.argtypes = [i64, i64]
    so.kurn_tbm_pack.restype = sz
    so.kurn_tbm_pack.argtypes = [ctypes.c_int, ctypes.c_char_p, i64, i64, vp, ctypes.POINTER(i64)]
    so.kurn_tbm_zeros.restype = i64
    so.kurn_tbm_zeros.argtypes = [ctypes.c_int, ctypes.c_char_p, i64, i64, i64]
    so.kurn_tbm_cols.argtypes = [vp, ctypes.c_char_p, sz, ctypes.POINTER(ctypes.c_float), i64, i64, i64, i64, i64]
    so.kurn_tbm_tq1_to_tq2.argtypes = [ctypes.c_char_p, ctypes.c_char_p, i64]
    return so


@pytest.mark.parametrize("z", [0.0, 0.42, 0.9])
def test_c_packer_matches_reference(tbm_lib, z):
    rng = np.random.default_rng(11)
    n, k = 37, 768
    t, d = _trits(rng, n, k, z), _d(rng, n, k // 256)
    w = _tq2_0_bytes(t, d)
    cap = tbm_lib.kurn_tbm_bytes(k, n)
    buf = ctypes.create_string_buffer(cap)
    dense = ctypes.c_int64()
    used = tbm_lib.kurn_tbm_pack(35, w, k, n, ctypes.addressof(buf), ctypes.byref(dense))
    ref = T.encode(t, d)
    assert used == len(ref) <= cap
    assert buf.raw[:used] == ref
    assert dense.value == T.header(ref)["dense"]
    assert tbm_lib.kurn_tbm_zeros(35, w, k, n, 1) == int(np.count_nonzero(t == 0))
    # TQ2_0 code 3 is not ternary: the packer refuses
    bad = bytearray(w)
    bad[0] |= 3
    assert tbm_lib.kurn_tbm_pack(35, bytes(bad), k, n, ctypes.addressof(buf), None) == 0


def test_c_tq1_to_tq2(tbm_lib):
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    rng = np.random.default_rng(13)
    t = _trits(rng, 4, 512, 0.35)
    t[:, ::256] = -1
    q1 = quantize(t.astype(np.float32) * 0.05, GGMLQuantizationType.TQ1_0).tobytes()
    out = ctypes.create_string_buffer(8 * 66)
    tbm_lib.kurn_tbm_tq1_to_tq2(q1, out, 8)
    t2, _ = T.tq2_0_trits(out.raw, 8)
    assert (t2.reshape(4, 512) == t).all()
    assert out.raw[64:66] == q1[52:54]


def test_c_kernels_match_tq2_0_reference(tbm_lib):
    """GEMV and 2..8 columns against the TQ2_0 reference; every column equals the GEMV bit for bit."""
    rng = np.random.default_rng(17)
    prng = random.Random(17)
    n, k = 45, 512
    t, d = _trits(rng, n, k, 0.42), _d(rng, n, k // 256)
    w = _tq2_0_bytes(t, d)
    buf = ctypes.create_string_buffer(tbm_lib.kurn_tbm_bytes(k, n))
    assert tbm_lib.kurn_tbm_pack(35, w, k, n, ctypes.addressof(buf), None) > 0
    M = 8
    X = q8_K_blocks(prng, M * k // 256)
    xrow = 292 * (k // 256)
    ref = reference_gemm("tq2_0", w, X, k, n, M)
    y1 = (ctypes.c_float * n)()
    tbm_lib.kurn_tbm_cols(ctypes.addressof(buf), X, xrow, y1, k, n, 1, 0, n)
    for r in range(n):
        assert abs(y1[r] - ref[r]) <= 1e-5 * max(1.0, abs(ref[r]))
    for m in range(2, M + 1):
        Y = (ctypes.c_float * (n * m))()
        tbm_lib.kurn_tbm_cols(ctypes.addressof(buf), X, xrow, Y, k, n, m, 0, n)
        for j in range(m):
            for r in range(n):
                assert abs(Y[j * n + r] - ref[j * n + r]) <= 1e-5 * max(1.0, abs(ref[j * n + r]))
        assert list(Y[:n]) == list(y1)
    # a row range that starts at a 16-row group and ends inside one: rows outside are not written
    Y = (ctypes.c_float * n)(*([7.0] * n))
    tbm_lib.kurn_tbm_cols(ctypes.addressof(buf), X, xrow, Y, k, n, 1, 16, 37)
    assert list(Y[16:37]) == list(y1[16:37]) and Y[15] == 7.0 and Y[37] == 7.0
