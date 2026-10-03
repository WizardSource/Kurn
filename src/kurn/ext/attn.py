"""WS-G: attention (`op attn`, kurn.attention) and its `kurn attn` command. kurn.attention is
imported lazily: it builds on kurn.spec/toolchain, which import this package."""

from .. import hooks


def _attn_cli(argv):
    from ..attention.cli import main

    return main(argv)


hooks.OPS["attn"] = "kurn.attention"
hooks.COMMANDS["attn"] = _attn_cli
