"""k4c keys as a llama.cpp KV cache type (integration/llama.cpp/k4c): with a built checkout that has apply.sh and
k4c/apply.sh applied (KURN_LLAMA_CPP), run k4c/test_k4c.c: K4C cache writes in llama.cpp's patterns (prefill ending
inside a group, one-row appends, rollback, out-of-order rows), clears, kurn's FLASH_ATTN_EXT on the K4C cache against
ggml on the dequantized keys, and batch invariance in exact mode."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT, require_tree

require_tree("integration/llama.cpp/k4c/test_k4c.c")
K4C = ROOT / "integration" / "llama.cpp" / "k4c"


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    for b in ("build", "build-k4c"):
        lib = d / b / "bin" / "libggml-cpu.so"
        if (d / "ggml" / "src" / "ggml-k4c.c").exists() and lib.exists() and "ggml_k4c_update_group" in open(lib.parent / "libggml-base.so", "rb").read().decode("latin-1"):
            return d, d / b / "bin"
    return None


def test_patch_and_apply_script_are_consistent():
    patch = (K4C / "llama-k4c.patch").read_text()
    for f in ("ggml/include/ggml.h", "ggml/src/ggml.c", "ggml/src/ggml-cpu/ops.cpp", "ggml/src/ggml-cpu/ggml-cpu.cpp",
              "src/llama-kv-cache.cpp", "src/llama-context.cpp", "common/arg.cpp", "tools/llama-bench/llama-bench.cpp"):
        assert f"+++ b/{f}" in patch
    assert "GGML_TYPE_K4C" in patch and "ggml_set_rows_k4c" in patch and "ggml-k4c.c" in patch
    assert "ggml_k4c_update_group" in (K4C / "ggml-k4c.c").read_text()


@pytest.mark.skipif(_llama_dir() is None, reason="no built llama.cpp checkout with k4c/apply.sh applied (KURN_LLAMA_CPP)")
@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_k4c_cache_and_attention(tmp_path):
    d, bindir = _llama_dir()
    exe = tmp_path / "test_k4c"
    subprocess.run(["gcc", "-O2", f"-I{d}/ggml/include", str(K4C / "test_k4c.c"), f"-L{bindir}", "-lggml", "-lggml-base",
                    "-lggml-cpu", "-lm", f"-Wl,-rpath,{bindir}", "-o", str(exe)], check=True)  # fmt: skip
    env = dict(os.environ, GGML_KURN_AMX="0")
    r = subprocess.run([str(exe), "4"], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "0 failed" in r.stdout
    r = subprocess.run([str(exe), "invariance"], capture_output=True, text=True, timeout=600, env=dict(env, GGML_KURN_FA_MODE="exact"))
    assert r.returncode == 0, r.stdout[-2000:]
