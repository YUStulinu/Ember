#include "backend/cpu/cpu_kernels.hpp"

#include <algorithm>
#include <vector>

#include "common/error.hpp"
#include "common/fp16.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace ember::cpu {

Weight Weight::from(const TensorView &t) {
    Weight w;
    w.dtype = t.dtype;
    w.data = t.data;
    if (t.shape.size() == 1) {
        w.rows = 1;
        w.cols = t.shape[0];
    } else if (t.shape.size() == 2) {
        w.rows = t.shape[0];
        w.cols = t.shape[1];
    } else {
        fail("tensor {} has {} dimensions; expected 1 or 2", t.name, t.shape.size());
    }
    if (w.dtype != DType::f32 && w.dtype != DType::f16 && w.dtype != DType::bf16)
        fail("tensor {} has dtype {}; the CPU backend reads F32, F16 and BF16", t.name, dtype_name(w.dtype));
    return w;
}

bool has_avx2() {
    static const bool ok = [] {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int r[4];
        __cpuid(r, 1);
        bool fma = r[2] & (1 << 12), f16c = r[2] & (1 << 29), osxsave = r[2] & (1 << 27);
        if (!osxsave) return false;
        if ((_xgetbv(0) & 6) != 6) return false;  // OS saves YMM registers
        __cpuidex(r, 7, 0);
        return fma && f16c && (r[1] & (1 << 5));
#elif defined(__x86_64__) || defined(__i386__)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
        return false;
#endif
    }();
    return ok;
}

void load_row(const Weight &w, int64_t r, float *out) {
    const int64_t n = w.cols;
    switch (w.dtype) {
        case DType::f32: std::copy_n(static_cast<const float *>(w.data) + r * n, n, out); break;
        case DType::bf16: {
            const uint16_t *src = static_cast<const uint16_t *>(w.data) + r * n;
            if (has_avx2()) avx2::widen_bf16(src, out, n);
            else for (int64_t i = 0; i < n; i++) out[i] = bf16_to_float(src[i]);
            break;
        }
        case DType::f16: {
            const uint16_t *src = static_cast<const uint16_t *>(w.data) + r * n;
            if (has_avx2()) avx2::widen_f16(src, out, n);
            else for (int64_t i = 0; i < n; i++) out[i] = fp16_to_float(src[i]);
            break;
        }
        default: fail("unsupported weight dtype");
    }
}

float dot(const float *a, const float *b, int n) {
    if (has_avx2()) return avx2::dot(a, b, n);
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void matmul(ThreadPool &pool, const float *x, int T, const Weight &w, float *y) {
    const int K = static_cast<int>(w.cols);
    const int64_t N = w.rows;
    // Work in blocks of 4 output rows: widen them once, then reuse them for
    // every token, so the widening cost is shared across the batch.
    const int64_t blocks = (N + 3) / 4;
    pool.parallel_for(static_cast<size_t>(blocks), 1, [&](size_t b0, size_t b1) {
        thread_local std::vector<float> rows;
        rows.resize(static_cast<size_t>(4) * K);
        for (size_t b = b0; b < b1; b++) {
            int64_t n0 = static_cast<int64_t>(b) * 4;
            int nr = static_cast<int>(std::min<int64_t>(4, N - n0));
            for (int i = 0; i < nr; i++) load_row(w, n0 + i, rows.data() + static_cast<size_t>(i) * K);
            for (int i = nr; i < 4; i++) std::fill_n(rows.data() + static_cast<size_t>(i) * K, K, 0.0f);
            const float *r0 = rows.data(), *r1 = r0 + K, *r2 = r1 + K, *r3 = r2 + K;
            for (int t = 0; t < T; t++) {
                const float *xt = x + static_cast<size_t>(t) * K;
                float out[4];
                if (has_avx2()) {
                    avx2::dot4(r0, r1, r2, r3, xt, K, out);
                } else {
                    for (int i = 0; i < 4; i++) {
                        float s = 0;
                        const float *r = rows.data() + static_cast<size_t>(i) * K;
                        for (int k = 0; k < K; k++) s += r[k] * xt[k];
                        out[i] = s;
                    }
                }
                float *yt = y + static_cast<size_t>(t) * N + n0;
                for (int i = 0; i < nr; i++) yt[i] = out[i];
            }
        }
    });
}

}  // namespace ember::cpu
