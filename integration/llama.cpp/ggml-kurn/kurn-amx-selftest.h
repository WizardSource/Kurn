#pragma once
// AMX permission and startup self-test shared by the KURN buffer type and kurn attention (kurn-amx-selftest.cpp).

enum kurn_amx_state {
    KURN_AMX_UNAVAILABLE = 0,  // no XTILEDATA permission, or AMX turned off by the self-test
    KURN_AMX_OK          = 1,  // permitted; the self-test passed or did not run
    KURN_AMX_GUARDED     = 2,  // permitted, but the self-test saw tile data lost across a context switch:
                               // AMX only with the preemption guard, nothing handed to ggml's unguarded AMX buffer
};

// Requests ARCH_REQ_XCOMP_PERM (XTILEDATA) and runs the self-test, once per process; later calls return the
// cached result. Thread-safe.
kurn_amx_state kurn_amx_check();
