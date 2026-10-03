#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// KURN extra buffer type: weights of the formats in kurn_dispatch.h are repacked at load
// into kurn's interleaved layout (the only copy) and every MUL_MAT / MUL_MAT_ID on them
// runs kurn kernels. NULL when the build lacks AVX-512 VNNI or GGML_KURN=0.
ggml_backend_buffer_type_t ggml_backend_cpu_kurn_buffer_type(void);

#ifdef __cplusplus
}
#endif
