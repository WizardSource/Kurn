"""Cost-aware speculative verify width: the `kurn specwidth` command (kurn.specwidth)."""

from .. import hooks


def _specwidth_cli(argv):
    from ..specwidth import main

    return main(argv)


hooks.COMMANDS["specwidth"] = _specwidth_cli
