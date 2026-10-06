"""Hybrid device routing: CPU-only, GPU-only, or CPU and GPU working together.

`route()` picks a device for one matmul. `plan()` assigns a whole decode/prefill
step so both devices can run in the same step when that is profitable.
"""

from .route import MODES, plan, route

__all__ = ["MODES", "plan", "route"]
