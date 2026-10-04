"""CUDA backend (`target cuda`, kurn.gpu): the `kurn gpu` command and routing of spec commands
(`kurn check|gen|build|verify|tune SPEC`) for specs whose target is cuda. kurn.gpu is imported
lazily: it builds on kurn.spec, which imports this package."""

from .. import hooks


def _gpu_cli(argv):
    from ..gpu.cli import main

    return main(argv)


def _spec_command(cmd, argv):
    from ..gpu.cli import spec_command

    return spec_command(cmd, argv)


hooks.COMMANDS["gpu"] = _gpu_cli
hooks.TARGET_BACKENDS["cuda"] = _spec_command
