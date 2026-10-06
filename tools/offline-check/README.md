# kurn offline check

One command that checks this copy of kurn on the machine it runs on: correctness of every kernel the CPU can run
(`kurn verify --all --strict`), measured read bandwidth, short energy-ranked tune sweeps, and an AMX tile-state test. Use it
on hosts that kurn has not been measured on (AMD Zen, AVX2-only desktops, real ARM, bare-metal AMX, WSL2).

```sh
tools/offline-check/run_offline_check.sh            # 15-45 min (tune sweeps rank on medians of 3 interleaved rounds)
QUICK=1 tools/offline-check/run_offline_check.sh    # 5-minute smoke test (example specs only)
FULL=1 tools/offline-check/run_offline_check.sh     # also the full pytest suite
KURN_CC=clang tools/offline-check/run_offline_check.sh   # pick the C compiler
```

Needs Linux, python3 ≥ 3.9 and gcc or clang. It installs nothing outside the output directory (a venv, a build cache and the
results). Without network access, it runs kurn straight from the source tree. Close other heavy programs while it runs,
because timings are recorded.

Output: `kurn-offline-results-<host>-<date>.tar.gz` in the current directory (or `$KURN_OFFLINE_OUT`).

The AMX test runs only on CPUs with AMX (Xeon Sapphire Rapids or newer). On a correct host every row reads "0 wrong". The
8-vCPU KVM guest kurn was developed on does not pass it: that guest loses AMX tile data across context switches.
