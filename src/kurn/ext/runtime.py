"""WS-F runtime: extra barrier wait policies for the tuner's `wait` key.

The bundled harness (`data/bench.c`) understands them through its `--wait` flag:
`futex` sleeps in the kernel at once, `hybrid:N` spins N pause instructions first
(PAUSE measured at ~16 ns on this Emerald Rapids VM, so hybrid:2000 ~ 30 us) and then sleeps. Tune them against a modelled
serial gap, e.g. `kurn tune spec.kurn wait=spin,hybrid:2000,futex --bench-args "--serial-us 20"`.
UMONITOR/UMWAIT waits exist in the runtime (`kurn/runtime/kurn_rt.h`) but not in
bench.c's barrier, so they are not offered here.
"""

from .. import hooks

WAITS = ("futex", "hybrid:2000", "hybrid:20000")


def _waits(op, f, t):
    return WAITS


hooks.extra_values("wait", _waits)
