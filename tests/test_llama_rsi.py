"""RSI build for llama-server (integration/llama.cpp/rsi): retrieval drafts as an arm of the cost-aware selector,
probabilities for accepted speculative tokens, run manifest + probability hash chain, position-keyed sampling.

Static checks on the patch always run; with a checkout that has rsi/apply.sh applied (KURN_LLAMA_CPP) the patch
must reverse-apply to it, its headers must be the ones shipped here, and kurn-replay.h's hash chain must match an
independent implementation of the documented format."""

import hashlib
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT, require_tree

require_tree("integration/llama.cpp/rsi/llama-rsi.patch")
RSI = ROOT / "integration" / "llama.cpp" / "rsi"


def test_patch_hooks():
    p = (RSI / "llama-rsi.patch").read_text()
    for f in ("common/speculative.cpp", "common/speculative.h", "common/sampling.cpp", "common/common.h",
              "tools/server/server-context.cpp", "tools/server/server-task.h", "tools/server/server-schema.cpp"):
        assert f"+++ b/{f}" in p, f
    for needle in ("allow_impl", "on_draft", "lazy_sync", "catch_up", "common_speculative_type_is_lookup",
                   "kurn::arm_selector", "take_retrieval", "allow_model", "kurn_spec_probs", "kurn_fill_probs",
                   "common_sampler_clone(smpl_save.get())", '"kurn_replay"', '"kurn_rng_position"', "kurn_dist_pos",
                   "kurn_position_uniform", "kurn_manifest_init", "KURN_SPEC_LAZY", "KURN_MANIFEST_SHA256"):
        assert needle in p, needle
    # the verify path no longer leaves accepted tokens without probabilities
    assert "+                // TODO: set result.probs" not in p
    a = (RSI / "apply.sh").read_text()
    assert "llama-rsi.patch" in a and "kurn-replay.h" in a and "kurn-build-info.h" in a and "arm_selector" in a


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    return d if (d / "tools" / "server" / "kurn-replay.h").exists() else None


@pytest.mark.skipif(_llama_dir() is None, reason="no llama.cpp checkout with rsi/apply.sh applied (KURN_LLAMA_CPP)")
def test_patch_matches_checkout():
    d = _llama_dir()
    r = subprocess.run(["git", "-C", str(d), "apply", "--check", "-R", str(RSI / "llama-rsi.patch")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (d / "tools" / "server" / "kurn-replay.h").read_text() == (RSI / "kurn-replay.h").read_text()
    sw = ROOT / "integration" / "llama.cpp" / "spec-width" / "kurn-spec-width.h"
    assert (d / "tools" / "server" / "kurn-spec-width.h").read_text() == sw.read_text()


CHAIN_DRIVER = r"""
#include "kurn-replay.h"
#include <iostream>
int main() {
    kurn::hash_chain c;
    c.init({151644, 872, 198, 13}, true);
    std::cout << c.hex() << "\n";
    c.add(785, 0.75f, {{785, 0.75f}, {576, 0.125f}});
    c.add(-1, 1.0f, {});
    c.add(42, 3.0e-8f, {{7, 0.5f}});
    std::cout << c.hex() << "\n" << c.n_tokens() << "\n";
    for (const auto & s : c.per_token()) std::cout << s << "\n";
}
"""


def _py_chain():
    def i32(v):
        return struct.pack("<I", v & 0xFFFFFFFF)

    def f32(v):
        return struct.pack("<f", v)

    h = hashlib.sha256(b"kurn-replay-v1" + b"".join(i32(t) for t in (151644, 872, 198, 13))).digest()
    out = [h.hex()]
    per = []
    for tok, p, top in ((785, 0.75, [(785, 0.75), (576, 0.125)]), (-1, 1.0, []), (42, 3.0e-8, [(7, 0.5)])):
        h = hashlib.sha256(h + i32(tok) + f32(p) + b"".join(i32(i) + f32(q) for i, q in top)).digest()
        per.append(h.hex()[:16])
    return out + [h.hex(), "3"] + per


@pytest.mark.skipif(_llama_dir() is None or not shutil.which("g++") or not shutil.which("gcc"),
                    reason="needs the llama.cpp checkout (vendor/hash) and gcc/g++")
def test_hash_chain_format(tmp_path):
    d = _llama_dir()
    v = d / "vendor"
    (tmp_path / "drv.cpp").write_text(CHAIN_DRIVER)
    subprocess.run(["gcc", "-c", "-O1", "-w", f"-I{v / 'hash'}", str(v / "hash" / "sha256" / "sha256.c"), "-o", str(tmp_path / "sha256.o")], check=True)
    subprocess.run(["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{RSI}", f"-I{v}", f"-I{v / 'hash'}",
                    str(tmp_path / "drv.cpp"), str(v / "hash" / "hash.cpp"), str(tmp_path / "sha256.o"), "-o", str(tmp_path / "drv")], check=True)
    got = subprocess.run([str(tmp_path / "drv")], capture_output=True, text=True, check=True).stdout.split()
    assert got == _py_chain()
