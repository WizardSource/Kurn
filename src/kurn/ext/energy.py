"""Energy counters: the `kurn energy` command (kurn.energy)."""

from .. import hooks


def _energy_cli(argv):
    from ..energy import main

    return main(argv)


hooks.COMMANDS["energy"] = _energy_cli
