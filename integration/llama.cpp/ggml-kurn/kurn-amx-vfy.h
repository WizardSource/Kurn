#pragma once
// AMX verify kernel for the KURN buffer type (kurn-amx-vfy.cpp).
#include <cstddef>
#include <cstdint>

// Y[m * N + n] for n in [r0, r1), m < M (2..16): Q8_0 weights in kurn's i16 records (buf, K values per
// row), X: M rows of K/32 block_q8_0 (row stride xrow); dx / nc: per row and block, d_x as float and
// -128 * sum(x) (row stride K/32). ng: row groups per pass (1..3). Bit-identical to the GEMV per
// column. Returns false (nothing computed) when AMX is not compiled in or M is out of range.
bool kurn_amx_vfy_q8_0(const uint8_t * buf, int64_t K, const char * X, size_t xrow, const float * dx, const int32_t * nc,
                       float * Y, int64_t N, int64_t M, int64_t r0, int64_t r1, int ng);
