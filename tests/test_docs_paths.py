"""Every file and directory README.md and CONTRIBUTING.md point to must exist in this copy of kurn, so a release
archive is self-contained: relative links, backticked paths (`tests/golden/`), bare file names (`formats.py`), and the
entries of the README's repository-layout block (`benchmarks/v0.2/ ... final/`)."""

import re

import pytest

from conftest import ROOT

DOCS = ("README.md", "CONTRIBUTING.md")
FILE_EXT = r"\.(?:py|c|h|cpp|inc|sh|kurn|md|toml|patch|csv|txt)"
LINK = re.compile(r"\]\(([^)\s]+)\)")
TICK = re.compile(r"`([^`\s]+)`")
LAYOUT_LINE = re.compile(r"^([\w.-]+(?:/[\w.-]+)*/)\s+(.*)$")


def _glob(rel):
    """Brace ranges ({1..4}) become wildcards; returns whether anything matches under ROOT."""
    pattern = re.sub(r"\{[^}]*\}", "*", rel.rstrip("/"))
    if any(ch in pattern for ch in "*?["):
        return any(ROOT.glob(pattern))
    return (ROOT / pattern).exists()


def _is_path(tok):
    if tok.startswith(("-", "http", "$", "~", "<", "/")) or "=" in tok or "<" in tok:
        return False
    return "/" in tok or re.fullmatch(rf"[\w.-]+{FILE_EXT}", tok) is not None


def _named_files():
    return {p.name for p in ROOT.rglob("*") if ".git" not in p.parts and "__pycache__" not in p.parts}


def references(doc):
    """(kind, path, base) for every local path the document mentions."""
    text = (ROOT / doc).read_text()
    out = []
    for target in LINK.findall(text):
        target = target.split("#", 1)[0]
        if target and not re.match(r"[a-z]+:", target):
            out.append(("link", target, ""))
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    for tok in TICK.findall(prose):
        tok = tok.split("::", 1)[0].rstrip(".,:;")
        if _is_path(tok):
            out.append(("path", tok, ""))
    layout = re.search(r"## Repository layout\s+```text\n(.*?)```", text, re.S)
    if layout:
        base = ""
        for line in layout.group(1).splitlines():
            m = LAYOUT_LINE.match(line)
            if m:
                base, desc = m.group(1), m.group(2)
                out.append(("layout", base, ""))
            else:
                desc = line.strip()
            for tok in re.findall(rf"[\w.-]+(?:/[\w.-]+)*(?:/|{FILE_EXT})(?=[\s,)]|$)", desc):
                out.append(("layout-entry", tok, base))
    return out


@pytest.mark.parametrize("doc", DOCS)
def test_doc_paths_exist(doc):
    names = _named_files()
    missing = []
    for kind, rel, base in references(doc):
        if kind == "layout-entry":  # relative to the line's directory or one of its parents (`data/x.c` under src/kurn/ext/)
            bases = [ROOT / base, *(ROOT / base).parents]
            ok = any(any(b.rglob(rel.rstrip("/"))) for b in bases if ROOT in b.parents)
        elif "/" not in rel.rstrip("/") and kind == "path":
            ok = rel in names or _glob(rel)
        else:
            ok = _glob(rel)
        if not ok:
            missing.append(f"{kind}: {base}{rel}" if base else f"{kind}: {rel}")
    assert not missing, f"{doc} references paths missing from this copy of kurn:\n  " + "\n  ".join(missing)


def test_finds_the_references_that_went_missing_before():
    refs = {(k, p) for k, p, _ in references("README.md")}
    assert ("layout-entry", "final/") in refs
    assert ("path", "tools/offline-check/run_offline_check.sh") in refs
    assert ("layout", "benchmarks/v0.2/") in refs
