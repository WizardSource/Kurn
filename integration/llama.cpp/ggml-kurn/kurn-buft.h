#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// KURN extra buffer type: weights of the formats in kurn_dispatch.h are repacked at load
// into kurn's interleaved layout (the only copy) and every MUL_MAT / MUL_MAT_ID on them
// runs kurn kernels. NULL when the build lacks AVX-512 VNNI or GGML_KURN=0.
ggml_backend_buffer_type_t ggml_backend_cpu_kurn_buffer_type(void);

// Fused group of MUL_MATs on KURN weights that share src1 (ffn gate + up, attention q + k + v), computed as one
// op; with glu (a SWIGLU of ops[0] = gate and ops[1] = up) only the SwiGLU result is written. Returns false,
// having done nothing, when the group is not supported; every thread gets the same answer.
struct ggml_compute_params;
struct ggml_tensor;
bool ggml_kurn_fused_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor ** ops, int n_ops,
                             struct ggml_tensor * glu);

// Compute thread count for ggml-cpu's backend (ggml_backend_cpu_set_n_threads): n_threads, or less after a failed
// AMX self-test when the threads would oversubscribe this process's vCPUs (kurn-amx-selftest.cpp).
int ggml_kurn_cap_threads(int n_threads);

// Whether ggml-cpu may register ggml's own AMX buffer type (apply.sh patches the extra buffer list): false with
// GGML_KURN_AMX=0 or when kurn's AMX self-test turned AMX off, so that "no AMX" means no AMX anywhere, and the small
// weights kurn's per-tensor fallback would leave with ggml stay on kurn's non-AMX kernels. True when the KURN buffer
// type is disabled (GGML_KURN=0).
bool ggml_kurn_ggml_amx_allowed(void);

#ifdef __cplusplus
}
#endif
