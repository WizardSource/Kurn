"""`op attn` spec handling, code generation and the `kurn attn` command."""

import pytest

from kurn import attention as A
from kurn import hooks
from kurn.cli import main
from kurn.spec import SpecError

from conftest import ROOT

SPECS = sorted((ROOT / "benchmarks" / "v0.2" / "attn").glob("*.kurn"))


def test_defaults_and_derived_dv():
    c = A.resolve({"target": "amx_bf16"})
    assert (c["op"], c["kv"], c["dk"], c["dv"], c["tile_q"], c["tile_kv"], c["split"], c["dec_rows"]) == \
        ("attn", "f16", 128, 128, 64, 128, 0, 8)  # fmt: skip
    assert A.resolve({"target": "avx512", "dk": 576})["dv"] == 512


@pytest.mark.parametrize(
    "extra,msg",
    [
        ({"kv": "q4_0"}, "kv='q4_0' not allowed"),
        ({"dk": 96}, "dk=96 not allowed"),
        ({"dk": 128, "dv": 64}, "dv=64 not allowed"),
        ({"tile_kv": 32}, "tile_kv=32 not allowed"),
        ({"rows": 4}, "unknown key 'rows'"),
        ({"heads": 6, "kv_heads": 4}, "multiple of kv_heads"),
        ({"target": "neon"}, "target 'neon'"),
        ({"op": "gemv"}, "only handles `op attn`"),
    ],
)
def test_resolve_errors(extra, msg):
    with pytest.raises(SpecError, match=msg):
        A.resolve({"target": "avx512", **extra})


def test_generate_sets_configuration():
    src = A.generate(A.resolve({"target": "avx512_bf16", "kv": "q8_0", "dk": 64, "tile_q": 32, "tile_kv": 256, "split": 4}))
    for line in (
        "#define KA_DK 64",
        "#define KA_DV 64",
        "#define KA_KV 2",
        "#define KA_ENGINE 1",
        "#define KA_TQ 32",
        "#define KA_TK 256",
        "#define KA_SPLIT 4",
        "#define KA_DEC_ROWS 8",
    ):
        assert line in src
    assert "void kattn(" in src and src == A.generate(A.resolve({"target": "avx512_bf16", "kv": "q8_0", "dk": 64,
                                                                  "tile_q": 32, "tile_kv": 256, "split": 4}))  # fmt: skip


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_example_specs_resolve(path):
    spec, space = A.load(path)
    c = A.resolve(spec)
    assert c["op"] == "attn"
    for k, vs in space.items():
        for v in vs:
            A.resolve(spec, {k: v})


def test_registered_as_extension_op():
    assert hooks.OPS["attn"] == "kurn.attention" and "attn" in hooks.COMMANDS


def test_cli_check_and_gen(capsys, tmp_path):
    spec = SPECS[0]
    assert main(["attn", "check", str(spec), "tile_q=32"]) == 0
    assert '"tile_q": 32' in capsys.readouterr().out
    out = tmp_path / "a.c"
    assert main(["attn", "gen", str(spec), "-o", str(out)]) == 0
    assert "#define KA_ENGINE 2" in out.read_text()
    assert main(["attn", "check", str(spec), "tile_q=33"]) == 2
    assert "tile_q=33 not allowed" in capsys.readouterr().err
    assert main(["attn", "check", str(spec), "tile_q=32,64"]) == 2


def test_core_cli_still_works(capsys):
    assert main(["targets"]) == 0
    assert "q8_0" in capsys.readouterr().out
