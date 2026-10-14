"""Workstream `ternary`: presence bitmap + packed signs for TQ1_0 / TQ2_0 weights (see kurn.ternary)."""

from .. import hooks


def _ternary(argv):
    from .._numpy import have_numpy

    if not have_numpy():
        import sys

        print("kurn: `kurn ternary` needs numpy and gguf: pip install 'kurn[gguf]'", file=sys.stderr)
        return 2
    from ..ternary import cli_main

    return cli_main(argv)


hooks.COMMANDS["ternary"] = _ternary
