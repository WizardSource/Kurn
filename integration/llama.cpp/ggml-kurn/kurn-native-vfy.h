#pragma once
// Decode / verify kernels for native-layout (ggml block) Q6_K and Q5_K weights in the KURN buffer
// (kurn-native-vfy.cpp).
#include <cstddef>
#include <cstdint>

enum { KURN_NATIVE_Q5_K = 13, KURN_NATIVE_Q6_K = 14 };  // GGML_TYPE_Q5_K / GGML_TYPE_Q6_K

bool kurn_native_vfy_supported(int type);

// Y[m * N + n] for n in [r0, r1), m < M: W = rows of ggml blocks of `type` (row stride wrow, K values),
// X = M rows of block_q8_K (row stride xrow). A column's result does not depend on M or on the row
// range. Returns false (nothing computed) for other types or without AVX-512 BW/VL.
bool kurn_native_vfy(int type, const char * W, size_t wrow, const char * X, size_t xrow, float * Y, int64_t K, int64_t N,
                     int64_t M, int64_t r0, int64_t r1);
