#pragma once

#include "ggml.h"

struct ggml_compute_params;

#ifdef __cplusplus
extern "C" {
#endif

// kurn attention for GGML_OP_FLASH_ATTN_EXT (called first by ggml_compute_forward_flash_attn_ext).
// Returns false, on every thread alike, when kurn does not take the node; ggml's own kernel then runs.
bool ggml_kurn_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
