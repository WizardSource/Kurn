"""CUDA backend (`target cuda`, `kurn gpu ...`) and hybrid device routing.

Registers:
  hooks.COMMANDS["gpu"]           -> `kurn gpu ...`
  hooks.TARGET_BACKENDS["cuda"]   -> routes check|gen|build|verify|tune on cuda specs
  hooks.COMMANDS["hybrid"]        -> `kurn hybrid ...` (cpu / gpu / together)
"""

from .. import hooks


def _gpu_cli(argv):
    from ..gpu.cli import main

    return main(argv)


def _cuda_spec(cmd, argv):
    from ..gpu.cli import spec_command

    return spec_command(cmd, argv)


def _hybrid_cli(argv):
    from ..hybrid.cli import main

    return main(argv)


hooks.COMMANDS["gpu"] = _gpu_cli
hooks.COMMANDS["hybrid"] = _hybrid_cli
hooks.TARGET_BACKENDS["cuda"] = _cuda_spec
