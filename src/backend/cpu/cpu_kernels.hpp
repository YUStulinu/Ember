// CPU math for the reference backend. Weights stay in their file format
// (bf16/f16/f32, memory-mapped) and are widened to f32 a few rows at a time.
#pragma once

#include <cstdint>

#include "common/thread_pool.hpp"
#include "model/safetensors.hpp"

namespace ember::cpu {

struct Weight {
    DType dtype = DType::f32;
    const void *data = nullptr;
    int64_t rows = 0, cols = 0;  // [rows, cols], row-major (rows = output features)

    static Weight from(const TensorView &t);
};

bool has_avx2();

// Widens row `r` of w into out[cols].
void load_row(const Weight &w, int64_t r, float *out);

// y[t, n] = sum_k x[t, k] * w[n, k]  for t < T, n < w.rows; x is [T, w.cols].
void matmul(ThreadPool &pool, const float *x, int T, const Weight &w, float *y);

// Dot product of two f32 vectors.
float dot(const float *a, const float *b, int n);

// Kernels the AVX2 translation unit provides (selected at runtime).
namespace avx2 {
void widen_bf16(const uint16_t *src, float *dst, int64_t n);
void widen_f16(const uint16_t *src, float *dst, int64_t n);
// out[i] = dot(rows[i], x) for i < 4.
void dot4(const float *r0, const float *r1, const float *r2, const float *r3, const float *x, int n, float *out);
float dot(const float *a, const float *b, int n);
}  // namespace avx2

}  // namespace ember::cpu
