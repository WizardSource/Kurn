"""kvz, lossless entropy coding of the KV cache in llama.cpp session state (integration/llama.cpp/kvz): the codec's
round trips and damaged-input handling (test_kvz.cpp, built here), and with a built checkout that has kvz/apply.sh
applied (KURN_LLAMA_CPP) and a GGUF model (KURN_KVZ_MODEL, default ~/models/Qwen3-0.6B-Q8_0.gguf): exact state and
logits after restore for every cache type (test_kvz_state.cpp), and llama-server --cache-kvz slot files and RAM
prompt cache (server_kvz_test.py)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, require_tree

require_tree("integration/llama.cpp/kvz/llama-kvz.cpp")
KVZ = ROOT / "integration" / "llama.cpp" / "kvz"


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    for b in ("build",):
        lib = d / b / "bin" / "libllama.so"
        if (d / "src" / "llama-kvz.cpp").exists() and lib.exists() and b"llama_state_seq_save_file_ext" in lib.read_bytes():
            return d, d / b / "bin"
    return None


def _model():
    m = Path(os.environ.get("KURN_KVZ_MODEL", os.path.expanduser("~/models/Qwen3-0.6B-Q8_0.gguf")))
    return m if m.exists() else None


def test_patch_and_apply_script_are_consistent():
    patch = (KVZ / "llama-kvz.patch").read_text()
    for f in ("include/llama.h", "src/llama-io.h", "src/llama-context.cpp", "src/llama-kv-cache.cpp", "src/CMakeLists.txt",
              "common/arg.cpp", "common/common.h", "tools/server/server-context.cpp"):
        assert f"+++ b/{f}" in patch, f
    for needle in ("LLAMA_STATE_SEQ_FLAGS_KVZ", "llama_state_seq_save_file_ext", "LLAMA_KVZ_MARK", "kvz_encode", "kvz_decode",
                   '"--cache-kvz"', "prompt_save_kvz", "size_only"):
        assert needle in patch, needle
    a = (KVZ / "apply.sh").read_text()
    assert "llama-kvz.patch" in a and "llama-kvz.h" in a and "llama-kvz.cpp" in a


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_codec(tmp_path):
    exe = tmp_path / "test_kvz"
    subprocess.run(["g++", "-O2", "-std=c++17", str(KVZ / "test_kvz.cpp"), str(KVZ / "llama-kvz.cpp"), "-lpthread", "-o", str(exe)], check=True)
    r = subprocess.run([str(exe), "quick"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:]
    assert " 0 failed" in r.stdout


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_codec_sanitizers(tmp_path):
    exe = tmp_path / "test_kvz_asan"
    r = subprocess.run(["g++", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-std=c++17",
                        str(KVZ / "test_kvz.cpp"), str(KVZ / "llama-kvz.cpp"), "-lpthread", "-o", str(exe)], capture_output=True, text=True)  # fmt: skip
    if r.returncode != 0:
        pytest.skip("no sanitizer runtime: " + r.stderr[-300:])
    r = subprocess.run([str(exe), "quick"], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.skipif(_llama_dir() is None, reason="no built llama.cpp checkout with kvz/apply.sh applied (KURN_LLAMA_CPP)")
def test_patch_matches_checkout():
    d, _ = _llama_dir()
    r = subprocess.run(["git", "-C", str(d), "apply", "--check", "-R", str(KVZ / "llama-kvz.patch")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("llama-kvz.h", "llama-kvz.cpp"):
        assert (d / "src" / f).read_text() == (KVZ / f).read_text(), f


@pytest.mark.skipif(_llama_dir() is None or _model() is None, reason="needs a kvz llama.cpp build (KURN_LLAMA_CPP) and a model (KURN_KVZ_MODEL)")
def test_state_roundtrip(tmp_path):
    d, bindir = _llama_dir()
    exe = tmp_path / "test_kvz_state"
    subprocess.run(["g++", "-O2", "-std=c++17", f"-I{d}/include", f"-I{d}/ggml/include", str(KVZ / "test_kvz_state.cpp"), f"-L{bindir}",
                    "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{bindir}", "-o", str(exe)], check=True)  # fmt: skip
    # k4c state as native group records (default) and as f16 rows (LLAMA_K4C_STATE=f16, alignment-independent)
    for env in ({}, {"LLAMA_K4C_STATE": "f16"}):
        r = subprocess.run([str(exe), str(_model()), "700"], capture_output=True, text=True, timeout=3600, env={**os.environ, **env})
        assert r.returncode == 0, (env, r.stdout[-4000:])
        assert " 0 failed" in r.stdout, env


@pytest.mark.skipif(_llama_dir() is None or _model() is None, reason="needs a kvz llama.cpp build (KURN_LLAMA_CPP) and a model (KURN_KVZ_MODEL)")
def test_server():
    _, bindir = _llama_dir()
    r = subprocess.run([sys.executable, str(KVZ / "server_kvz_test.py"), str(bindir / "llama-server"), str(_model()), "k4c_q4", "f16"],
                       capture_output=True, text=True, timeout=3600)  # fmt: skip
    assert r.returncode == 0, r.stdout[-4000:]
