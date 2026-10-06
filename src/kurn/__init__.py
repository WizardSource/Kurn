"""kurn: a small spec language for quantized CPU inference kernels.

>>> import kurn
>>> spec, space = kurn.parse("op gemv\\nweights q8_0\\ntarget avx512_vnni\\nlayout vnni16")
>>> src = kurn.generate(kurn.resolve(spec))
"""

__version__ = "0.3.0.dev3"

from .formats import FORMATS
from .kernels import KERNELS, embed, generate
from .spec import SpecError, load, parse, resolve

__all__ = ["FORMATS", "KERNELS", "SpecError", "__version__", "embed", "generate", "load", "parse", "resolve"]
