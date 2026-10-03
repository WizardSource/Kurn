"""Workstream extensions. Every module in this package is imported once the core
registries exist (end of kurn.kernels), in name order. A module registers its
formats, recipes, kernels, lowerings and schedule values through kurn.hooks and
the registry dicts, so parallel work does not edit shared files."""

import importlib
import pkgutil

for _m in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    importlib.import_module(f"{__name__}.{_m.name}")
