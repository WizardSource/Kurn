"""Cost-aware verify width in llama-server (integration/llama.cpp/spec-width/llama-server-spec-width.patch): the patch
carries the server flags, the per-slot policy and the hooks around drafting and acceptance, and (with a checkout that
has spec-width/apply.sh applied, KURN_LLAMA_CPP) it is exactly what that checkout contains."""

import os
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT, require_tree

require_tree("integration/llama.cpp/spec-width/llama-server-spec-width.patch")
SW = ROOT / "integration" / "llama.cpp" / "spec-width"


def test_server_patch_hooks():
    p = (SW / "llama-server-spec-width.patch").read_text()
    for f in ("common/common.h", "common/arg.cpp", "tools/server/server-context.cpp"):
        assert f"+++ b/{f}" in p
    for needle in ('"--spec-width"', '"--spec-width-mode"', "KURN_SPEC_WIDTH", "kurn::width_policy", "keep_drafting",
                   "->truncate(", "->observe(", "->n_cap()", "verify widths (M:steps)"):
        assert needle in p, needle
    a = (SW / "apply.sh").read_text()
    assert "llama-server-spec-width.patch" in a and "tools/server/" in a


def _llama_dir():
    d = Path(os.environ.get("KURN_LLAMA_CPP", os.path.expanduser("~/src/llama-kurn")))
    return d if (d / "tools" / "server" / "kurn-spec-width.h").exists() else None


def _without_rsi(d, tmp):
    """A copy of the files the spec-width patches touch, with rsi/llama-rsi.patch (applied on top of spec-width
    when the checkout has rsi/apply.sh) reversed, so the spec-width patches can be checked on their own."""
    rsi = ROOT / "integration" / "llama.cpp" / "rsi" / "llama-rsi.patch"
    files = set()
    for p in ("llama-server-spec-width.patch", "llama-spec-width.patch"):
        files |= {line[6:].strip() for line in (SW / p).read_text().splitlines() if line.startswith("+++ b/")}
    for f in files:
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp / f).write_bytes((d / f).read_bytes())
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    if (d / "tools" / "server" / "kurn-replay.h").exists() and rsi.exists():
        inc = [a for f in sorted(files) for a in ("--include", f)]
        subprocess.run(["git", "-C", str(tmp), "apply", "-R", *inc, str(rsi)], check=True)
    return tmp


@pytest.mark.skipif(_llama_dir() is None, reason="no llama.cpp checkout with spec-width/apply.sh applied (KURN_LLAMA_CPP)")
def test_server_patch_matches_checkout(tmp_path):
    d = _without_rsi(_llama_dir(), tmp_path)
    r = subprocess.run(["git", "-C", str(d), "apply", "--check", "-R", str(SW / "llama-server-spec-width.patch")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (_llama_dir() / "tools" / "server" / "kurn-spec-width.h").read_text() == (SW / "kurn-spec-width.h").read_text()
