"""llama.cpp integration (kurn/integration/llama.cpp): the registry-driven kernel generator and
its packing glue, and (when a built checkout is available) the KURN buffer type checker.

The checker part needs a llama.cpp checkout with apply.sh applied and built; point
KURN_LLAMA_CPP at it (default ~/src/llama-kurn). It is skipped when absent."""

import ctypes
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kurn.formats import FORMATS, reference_gemm
from kurn.targets import TARGETS
from kurn.toolchain import cpu_flags

from conftest import BLOCKS, ROOT, require_tree

require_tree("integration/llama.cpp/gen_ggml_sources.py")
INTEG = ROOT / "integration" / "llama.cpp"
sys.path.insert(0, str(INTEG))
import gen_ggml_sources as gen  # noqa: E402

HAVE_VNNI512 = TARGETS["avx512_vnni"].requires <= cpu_flags()
FAKE_GGML_H = (
    "enum ggml_type {\n"
    + "".join(
        f"    GGML_TYPE_{n.upper()} = {i},\n" for i, n in enumerate(["q8_0", "q4_0", "iq4_nl", "q4_K", "q2_0", "tq2_0", "q1_0", "q8_K"])
    )
    + "};\n"
)


def _generate(tmp_path, *extra):
    h = tmp_path / "ggml.h"
    h.write_text(FAKE_GGML_H)
    out = tmp_path / "kurn"
    gen.main([str(out), "--ggml-h", str(h), *extra])
    return out


def test_discovers_every_registry_format_ggml_knows(tmp_path):
    fmts = gen.discover(gen.ggml_types(None))
    assert {"q8_0", "q4_0", "iq4_nl", "q4_K", "q2_0", "tq2_0", "q1_0"} <= set(fmts)
    out = _generate(tmp_path)
    disp = (out / "kurn_dispatch.h").read_text()
    for f in fmts:
        if f"GGML_TYPE_{f.upper()}" not in FAKE_GGML_H:
            continue
        assert (out / f"kurn_{f}_gemv.c").exists()
        for cols in gen.VFY_COLS:
            assert (out / f"kurn_{f}_vfy{cols}.c").exists()
        assert f'"{f}"' in disp
    # formats ggml lacks are skipped
    assert gen.discover({"GGML_TYPE_Q8_0"}) == ["q8_0"]


def test_config_overrides_and_verify_inherits_algorithm(tmp_path):
    import json

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"q4_0": {"rows": 4, "unpack": "lut"}}))
    out = _generate(tmp_path, "--config", str(cfg), "--only", "q4_0")
    disp = (out / "kurn_dispatch.h").read_text()
    assert "rows=4" in disp and "unpack=lut" in disp
    assert "LUT[16]" in (out / "kurn_q4_0_vfy8.c").read_text()  # verify uses the GEMV's unpack choice


@pytest.mark.skipif(not HAVE_VNNI512 or shutil.which("gcc") is None, reason="needs gcc and an AVX-512 VNNI CPU")
def test_packing_glue_gemv_and_verify_match_reference(tmp_path):
    """Pack into caller memory (as the buffer type does), then GEMV and 2-8 column verify
    against the Python reference; the verify columns must equal the GEMV bit for bit."""
    out = _generate(tmp_path)
    lib = tmp_path / "libk.so"
    srcs = sorted(str(p) for p in out.glob("kurn_*.c"))
    subprocess.run(
        ["gcc", "-O2", "-march=x86-64-v4", "-mavx512vnni", "-shared", "-fPIC", "-I", str(out), *srcs, "-o", str(lib)], check=True
    )
    so = ctypes.CDLL(str(lib))
    rng = random.Random(7)
    i64 = ctypes.c_int64
    for f in gen.discover(gen.ggml_types(str(tmp_path / "ggml.h"))):
        fmt = FORMATS[f]
        act = fmt.act
        K = max(fmt.block, FORMATS[act].block) * 2
        N, M = 37, 8
        W = BLOCKS[f](rng, N * K // fmt.block)
        X = BLOCKS[act](rng, M * K // FORMATS[act].block)
        entry = gen.kernel({"op": "gemv", "weights": f}).entry
        vbase = gen.kernel({"op": "verify", "weights": f}).entry
        so[f"{entry}_kurn_bytes"].restype = ctypes.c_size_t
        so[f"{entry}_kurn_view"].restype = ctypes.c_void_p
        nbytes = so[f"{entry}_kurn_bytes"](i64(K), i64(N))
        buf = ctypes.create_string_buffer(nbytes + 64)
        base = (ctypes.addressof(buf) + 63) & ~63
        so[f"{entry}_kurn_pack"](W, i64(K), i64(N), ctypes.c_void_p(base))
        pk = ctypes.c_void_p(so[f"{entry}_kurn_view"](ctypes.c_void_p(base), i64(K), i64(N)))
        ref = reference_gemm(f, W, X, K, N, M)
        xrow = FORMATS[act].row_bytes(K)
        y1 = (ctypes.c_float * N)()
        so[f"{entry}_packed"](pk, X, y1, i64(K), i64(0), i64(N))
        for n in range(N):
            assert abs(y1[n] - ref[n]) <= 1e-5 * max(1.0, abs(ref[n])), (f, n)
        for cols in gen.VFY_COLS:
            for m in (cols // 2 + 1, cols):
                Y = (ctypes.c_float * (N * m))()
                so[f"{vbase}{cols}_packed"](pk, X, Y, i64(K), i64(N), i64(m), i64(0), i64(N))
                for j in range(m):
                    for n in range(N):
                        assert abs(Y[j * N + n] - ref[j * N + n]) <= 1e-5 * max(1.0, abs(ref[j * N + n])), (f, cols, m, j, n)
                assert list(Y[:N]) == list(y1), (f, cols, m)  # same arithmetic as the GEMV
        assert xrow > 0


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    lib = d / "build" / "bin" / "libggml-cpu.so"
    if not (d / "ggml" / "src" / "ggml-cpu" / "kurn" / "kurn-buft.cpp").exists() or not lib.exists():
        return None
    return d


@pytest.mark.skipif(_llama_dir() is None, reason="no llama.cpp checkout with the kurn buffer type (KURN_LLAMA_CPP)")
@pytest.mark.skipif(not HAVE_VNNI512 or shutil.which("gcc") is None, reason="needs gcc and an AVX-512 VNNI CPU")
def test_buffer_type_checker_smoke(tmp_path):
    d = _llama_dir()
    exe = tmp_path / "test_kurn_buft"
    subprocess.run(
        [
            "gcc",
            "-O2",
            f"-I{d}/ggml/include",
            str(INTEG / "test_kurn_buft.c"),
            f"-L{d}/build/bin",
            "-lggml",
            "-lggml-base",
            "-lggml-cpu",
            "-lm",
            f"-Wl,-rpath,{d}/build/bin",
            "-o",
            str(exe),
        ],
        check=True,
    )
    # GGML_KURN_AMX=0: AMX tile state is not preserved across context switches on some VMs
    r = subprocess.run([str(exe), "smoke"], capture_output=True, text=True, timeout=1200, env=dict(os.environ, GGML_KURN_AMX="0"))
    if "not available" in r.stdout:
        pytest.skip("KURN buffer type not compiled in")
    assert r.returncode == 0, r.stdout[-4000:]
    # native-layout Q6_K / Q5_K (kurn-native-vfy.cpp): below the AMX-BF16 threshold, and at every width in exact
    # mode (no AMX tiles are used by either run: the native kernels are AVX-512 only)
    for env in ({"GGML_KURN_EXACT": "1", "GGML_KURN_AMX": "0"}, {}):
        r = subprocess.run([str(exe), "native"], capture_output=True, text=True, timeout=1200, env={**os.environ, **env})
        assert r.returncode == 0, (env, r.stdout[-4000:])
