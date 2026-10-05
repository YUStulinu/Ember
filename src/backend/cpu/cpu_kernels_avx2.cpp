// Compiled with AVX2 + FMA + F16C enabled; only called after a runtime check.
#include "backend/cpu/cpu_kernels.hpp"
#include "common/fp16.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>

namespace ember::cpu::avx2 {

void widen_bf16(const uint16_t *src, float *dst, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
    for (; i < n; i++) dst[i] = bf16_to_float(src[i]);
}

void widen_f16(const uint16_t *src, float *dst, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
    for (; i < n; i++) dst[i] = fp16_to_float(src[i]);
}

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

void dot4(const float *r0, const float *r1, const float *r2, const float *r3, const float *x, int n, float *out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), xv, a3);
    }
    float s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
    for (; i < n; i++) {
        s0 += r0[i] * x[i];
        s1 += r1[i] * x[i];
        s2 += r2[i] * x[i];
        s3 += r3[i] * x[i];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

float dot(const float *a, const float *b, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    float s = hsum(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

}  // namespace ember::cpu::avx2

#else  // not x86: has_avx2() is false, so these are never called

namespace ember::cpu::avx2 {
void widen_bf16(const uint16_t *, float *, int64_t) {}
void widen_f16(const uint16_t *, float *, int64_t) {}
void dot4(const float *, const float *, const float *, const float *, const float *, int, float *) {}
float dot(const float *, const float *, int) { return 0; }
}  // namespace ember::cpu::avx2

#endif
