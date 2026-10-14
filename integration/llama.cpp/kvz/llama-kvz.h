#pragma once

// kvz: lossless entropy coding of KV cache rows for session state (slot files, the server's RAM prompt cache).
//
// A block is n_rows cache rows of row_bytes each, in one of the layouts below. Each layout is split into symbol
// streams (f16 high / low bytes, Q8_0 / Q4_0 scale bytes and codes), stored column-major so that every column
// (channel) is one run. Each stream is coded with static rANS (12-bit frequencies, 4 interleaved states) under a
// per-column-group context; the encoder picks the group width, and plain or row-delta symbols, per stream by
// estimated cost, and stores a stream raw when coding would not pay. Decoding gives back the exact input bytes.

#include <cstddef>
#include <cstdint>

enum kvz_layout : uint8_t {
    KVZ_RAW     = 0, // stored
    KVZ_PLANES2 = 1, // 2-byte elements (f16, bf16)
    KVZ_Q8_0    = 2,
    KVZ_Q4_0    = 3,
    KVZ_PLANES4 = 4, // 4-byte elements (f32)
};

// worst-case encoded size of a block of raw_bytes
size_t kvz_bound(size_t raw_bytes);

// encode n_rows rows of row_bytes from src into dst (capacity kvz_bound(n_rows * row_bytes)); returns the size
size_t kvz_encode(kvz_layout layout, const uint8_t * src, size_t n_rows, size_t row_bytes, uint8_t * dst, int n_threads);

// decode a block written by kvz_encode with the same n_rows and row_bytes; false if src is not a valid block
bool kvz_decode(const uint8_t * src, size_t src_size, uint8_t * dst, size_t n_rows, size_t row_bytes, int n_threads);

// threads for kvz in llama.cpp: LLAMA_KVZ_THREADS, else the hardware threads
int kvz_default_threads();
