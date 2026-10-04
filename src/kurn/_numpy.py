"""numpy is an optional dependency (`pip install 'kurn[compress]'`).

Modules that need it (compressed weights, mixed precision, the latent-KV and engine helpers) do
`from ._numpy import np`, so importing them works without numpy and the first numpy call raises an
ImportError that names the extra to install.
"""

import importlib


class _LazyNumpy:
    def __getattr__(self, name):
        try:
            mod = importlib.import_module("numpy")
        except ImportError as e:
            raise ImportError("this part of kurn needs numpy: pip install 'kurn[compress]' (or pip install numpy)") from e
        value = getattr(mod, name)
        setattr(self, name, value)
        return value

    def __repr__(self):
        return "<lazy numpy>"


np = _LazyNumpy()


def have_numpy():
    try:
        importlib.import_module("numpy")
        return True
    except ImportError:
        return False
