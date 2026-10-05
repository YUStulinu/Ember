// Weight-only quantization: int8 (one scale per row) and int4 (groups of 128
// along K with a scale and a minimum each).
//
// Decode kernels follow gemv_tc's design (each thread loads 16 contiguous
// bytes of a weight row and feeds mma.sync directly), with two twists:
//   - The tensor cores multiply the integer codes themselves: -127..127 and
//     0..15 are exact in fp16, so the per-row (int8) or per-group (int4) scale
//     is applied to the accumulated sums, not to every weight.
//   - int8 -> fp16 and int4 -> fp16 use the "magic number" trick: put the code
//     in the mantissa of 1024.0 (0x6400) and subtract 1024 (+128 for int8),
//     two weights per instruction.
//
// int4 packing (chosen so extraction is one shift and one mask): each 32-bit
// word holds 8 consecutive k; nibble j holds k = 2j and nibble 4 + j holds
// k = 2j + 1, so (word >> 4j) & 0x000F000F is the pair (2j, 2j + 1). A row is
// K / 2 bytes; group g occupies bytes [64g, 64g + 64), and thread t of a group
// reads bytes [64g + 16t, 64g + 16t + 16) - 32 consecutive k.
//
// The int4 result for a group is s * sum(q * x) + m * sum(x): the second term
// uses per-token group sums of x, computed by group_sums_kernel first.
#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/kernels.cuh"

namespace ember::cuda {

namespace {

constexpr float kHalfMax = 65504.0f;
constexpr int QWARPS = 4;

__device__ __forceinline__ half to_half_sat(float v) { return __float2half_rn(fminf(fmaxf(v, -kHalfMax), kHalfMax)); }
__device__ __forceinline__ float silu(float g) { return g / (1.0f + __expf(-g)); }

__device__ __forceinline__ uint4 ld_stream(const void *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Two int8 (bytes lo, lo + 1 of w) as a half2 of exact integers.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t w, uint32_t selector) {
    const uint32_t biased = w ^ 0x80808080u;                      // b + 128 as unsigned
    uint32_t h = __byte_perm(biased, 0x64646464u, selector);     // [b, 0x64] per half: 1024 + (b + 128)
    half2 v = *reinterpret_cast<half2 *>(&h);
    v = __hsub2(v, __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t *>(&v);
}

// Pair j of a packed int4 word as a half2 of exact integers 0..15.
__device__ __forceinline__ uint32_t i4x2_to_h2(uint32_t w, int j) {
    uint32_t h = ((w >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
    half2 v = *reinterpret_cast<half2 *>(&h);
    v = __hsub2(v, __float2half2_rn(1024.0f));
    return *reinterpret_cast<uint32_t *>(&v);
}

template <Epilogue EP>
__device__ __forceinline__ void store_one(void *C, int ldc, int M, int N, int tok, int n, float v, float partner) {
    if constexpr (EP == Epilogue::silu_mul) {
        if ((n & 1) == 0 && tok < M && n + 1 < N)
            static_cast<half *>(C)[static_cast<size_t>(tok) * ldc + n / 2] = to_half_sat(silu(v) * partner);
    } else {
        if (tok >= M || n >= N) return;
        if constexpr (EP == Epilogue::store_f16) static_cast<half *>(C)[static_cast<size_t>(tok) * ldc + n] = to_half_sat(v);
        else if constexpr (EP == Epilogue::store_f32) static_cast<float *>(C)[static_cast<size_t>(tok) * ldc + n] = v;
        else static_cast<float *>(C)[static_cast<size_t>(tok) * ldc + n] += v;
    }
}

// Sums the warps' accumulators through shared memory, then warp 0 writes C.
// `scale_a`/`scale_b` multiply rows g and g + 8 (int8 row scales; 1 for int4).
template <int MT, Epilogue EP>
__device__ __forceinline__ void reduce_and_store(float (&acc)[MT][4], float (*red)[MT][4][32], int lane, int warp, int n0,
                                                 float scale_a, float scale_b, const int M, const int N, void *C, int ldc) {
    if (warp > 0) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int e = 0; e < 4; e++) red[warp - 1][mt][e][lane] = acc[mt][e];
    }
    __syncthreads();
    if (warp != 0) return;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            float v = acc[mt][e];
#pragma unroll
            for (int w = 0; w < QWARPS - 1; w++) v += red[w][mt][e][lane];
            v *= (e < 2) ? scale_a : scale_b;
            const float partner = __shfl_down_sync(0xffffffffu, v, 4);  // row n + 1 (silu_mul pairs)
            store_one<EP>(C, ldc, M, N, mt * 8 + 2 * t + (e & 1), n0 + g + (e >> 1) * 8, v, partner);
        }
}

// ---- int8 ---------------------------------------------------------------------------

template <int MT, Epilogue EP>
__global__ void __launch_bounds__(QWARPS * 32)
    gemv_q8_kernel(const half *__restrict__ A, int lda, const int8_t *__restrict__ Wq, const float *__restrict__ scales,
                   int M, int N, int K, void *__restrict__ C, int ldc) {
    __shared__ float red[QWARPS - 1][MT][4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    const int ra = min(n0 + g, N - 1), rb = min(n0 + g + 8, N - 1);
    const int8_t *wa = Wq + static_cast<size_t>(ra) * K + 16 * t;
    const int8_t *wb = Wq + static_cast<size_t>(rb) * K + 16 * t;
    const half *xr[MT];
    bool xok[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int tok = mt * 8 + g;
        xok[mt] = tok < M;
        xr[mt] = A + static_cast<size_t>(min(tok, M - 1)) * lda + 16 * t;
    }
    float acc[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;

    const int chunks = K / 64;  // 64 k per chunk: 4 threads x 16 bytes
#pragma unroll 2
    for (int c = warp; c < chunks; c += QWARPS) {
        const int kb = c * 64;
        const uint4 qa = ld_stream(wa + kb), qb = ld_stream(wb + kb);
        const uint32_t *ua = reinterpret_cast<const uint32_t *>(&qa), *ub = reinterpret_cast<const uint32_t *>(&qb);
        uint32_t af[8][2];
#pragma unroll
        for (int w = 0; w < 4; w++) {
            af[2 * w][0] = i8x2_to_h2(ua[w], 0x4140);
            af[2 * w + 1][0] = i8x2_to_h2(ua[w], 0x4342);
            af[2 * w][1] = i8x2_to_h2(ub[w], 0x4140);
            af[2 * w + 1][1] = i8x2_to_h2(ub[w], 0x4342);
        }
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
            if (xok[mt]) {
                x0 = *reinterpret_cast<const uint4 *>(xr[mt] + kb);
                x1 = *reinterpret_cast<const uint4 *>(xr[mt] + kb + 8);
            }
            const uint32_t *xa = reinterpret_cast<const uint32_t *>(&x0), *xb = reinterpret_cast<const uint32_t *>(&x1);
#pragma unroll
            for (int j = 0; j < 4; j++) mma_16x8x8(acc[mt], af[j][0], af[j][1], xa[j]);
#pragma unroll
            for (int j = 0; j < 4; j++) mma_16x8x8(acc[mt], af[4 + j][0], af[4 + j][1], xb[j]);
        }
    }
    reduce_and_store<MT, EP>(acc, red, lane, warp, n0, scales[ra], scales[rb], M, N, C, ldc);
}

// ---- int4 ---------------------------------------------------------------------------

template <int MT, Epilogue EP>
__global__ void __launch_bounds__(QWARPS * 32)
    gemv_q4_kernel(const half *__restrict__ A, int lda, const uint8_t *__restrict__ Wq, const half2 *__restrict__ sm,
                   const float *__restrict__ xsum, int M, int N, int K, void *__restrict__ C, int ldc) {
    __shared__ float red[QWARPS - 1][MT][4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16, groups = K / kInt4Group;
    const int ra = min(n0 + g, N - 1), rb = min(n0 + g + 8, N - 1);
    const uint8_t *wa = Wq + static_cast<size_t>(ra) * (K / 2) + 16 * t;
    const uint8_t *wb = Wq + static_cast<size_t>(rb) * (K / 2) + 16 * t;
    const half2 *sa = sm + static_cast<size_t>(ra) * groups, *sb = sm + static_cast<size_t>(rb) * groups;
    const half *xr[MT];
    bool xok[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int tok = mt * 8 + g;
        xok[mt] = tok < M;
        xr[mt] = A + static_cast<size_t>(min(tok, M - 1)) * lda + 32 * t;
    }
    float acc[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;

    for (int grp = warp; grp < groups; grp += QWARPS) {
        const uint4 qa = ld_stream(wa + grp * 64), qb = ld_stream(wb + grp * 64);
        const uint32_t *ua = reinterpret_cast<const uint32_t *>(&qa), *ub = reinterpret_cast<const uint32_t *>(&qb);
        const float2 fa = __half22float2(sa[grp]), fb = __half22float2(sb[grp]);  // (scale, min)
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const half *xp = xr[mt] + grp * kInt4Group;
#pragma unroll
            for (int w = 0; w < 4; w++) {
                uint4 x = xok[mt] ? *reinterpret_cast<const uint4 *>(xp + 8 * w) : make_uint4(0, 0, 0, 0);
                const uint32_t *xw = reinterpret_cast<const uint32_t *>(&x);
#pragma unroll
                for (int j = 0; j < 4; j++) mma_16x8x8(tmp, i4x2_to_h2(ua[w], j), i4x2_to_h2(ub[w], j), xw[j]);
            }
            // tmp holds sum(q * x) for rows (g, g + 8) and tokens (2t, 2t + 1) over this group.
            const int tok0 = min(mt * 8 + 2 * t, M - 1), tok1 = min(mt * 8 + 2 * t + 1, M - 1);
            const float xs0 = xsum[static_cast<size_t>(tok0) * groups + grp], xs1 = xsum[static_cast<size_t>(tok1) * groups + grp];
            acc[mt][0] += fa.x * tmp[0] + fa.y * xs0;
            acc[mt][1] += fa.x * tmp[1] + fa.y * xs1;
            acc[mt][2] += fb.x * tmp[2] + fb.y * xs0;
            acc[mt][3] += fb.x * tmp[3] + fb.y * xs1;
        }
    }
    reduce_and_store<MT, EP>(acc, red, lane, warp, n0, 1.0f, 1.0f, M, N, C, ldc);
}

// xsum[m, grp] = sum of x[m, grp * 128 .. grp * 128 + 127]; one warp per (m, grp).
__global__ void group_sums_kernel(const half *A, int lda, int M, int K, float *xsum) {
    const int groups = K / kInt4Group;
    const int item = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (item >= M * groups) return;
    const int m = item / groups, grp = item % groups;
    const half2 *p = reinterpret_cast<const half2 *>(A + static_cast<size_t>(m) * lda + grp * kInt4Group) + lane * 2;
    float2 a = __half22float2(p[0]), b = __half22float2(p[1]);
    float s = warp_sum(a.x + a.y + b.x + b.y);
    if (lane == 0) xsum[item] = s;
}

// ---- quantization and dequantization -------------------------------------------------

// int8: one block per row; scale = max|w| / 127.
__global__ void quantize_q8_kernel(const half *W, int K, int8_t *Wq, float *scales) {
    __shared__ float scratch[32];
    const int n = blockIdx.x;
    const half *row = W + static_cast<size_t>(n) * K;
    float mx = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(__half2float(row[k])));
    mx = block_max(mx, scratch);
    const float scale = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / scale;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        Wq[static_cast<size_t>(n) * K + k] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(__half2float(row[k]) * inv))));
    if (threadIdx.x == 0) scales[n] = scale;
}

// int4: one warp per (row, group). Each lane holds 4 consecutive k; lanes
// 2w and 2w + 1 together form one 8-k word of the packed layout.
//
// Plain min/max rounding wastes the 16 levels when a group has an outlier, so
// the range is searched: 25 clippings of [min, max] are tried, each refined by
// a least-squares fit of (scale, min) to its codes, and the one with the
// smallest squared error wins (the idea behind llama.cpp's k-quants).
__device__ __forceinline__ float warp_sum_all(float v) { return warp_sum(v); }

__device__ __forceinline__ void codes_for(const float (&v)[4], float s, float m, uint32_t (&q)[4]) {
    const float inv = 1.0f / s;
#pragma unroll
    for (int i = 0; i < 4; i++) q[i] = static_cast<uint32_t>(max(0, min(15, __float2int_rn((v[i] - m) * inv))));
}

__global__ void quantize_q4_kernel(const half *W, int N, int K, uint8_t *Wq, half2 *sm) {
    const int groups = K / kInt4Group;
    const int item = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (item >= N * groups) return;
    const int n = item / groups, grp = item % groups;
    const half *src = W + static_cast<size_t>(n) * K + grp * kInt4Group + lane * 4;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = __half2float(src[i]);
    float lo = fminf(fminf(v[0], v[1]), fminf(v[2], v[3])), hi = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
    for (int o = 16; o > 0; o >>= 1) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    const float range = hi - lo;
    float best_s = range > 0.0f ? range / 15.0f : 1.0f, best_m = lo, best_err = INFINITY;
    if (range > 0.0f) {
        const float clips[5] = {0.0f, 0.02f, 0.05f, 0.09f, 0.14f};
        for (int a = 0; a < 5; a++) {
            for (int b = 0; b < 5; b++) {
                const float l = lo + clips[a] * range, h = hi - clips[b] * range;
                float s = (h - l) / 15.0f, m = l;
                uint32_t q[4];
                codes_for(v, s, m, q);
                // Least squares for (s, m) given the codes: minimize sum (q s + m - w)^2.
                float sq = 0, sqq = 0, sw = 0, sqw = 0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float qf = static_cast<float>(q[i]);
                    sq += qf;
                    sqq += qf * qf;
                    sw += v[i];
                    sqw += qf * v[i];
                }
                sq = warp_sum_all(sq);
                sqq = warp_sum_all(sqq);
                sw = warp_sum_all(sw);
                sqw = warp_sum_all(sqw);
                const float cnt = static_cast<float>(kInt4Group), det = cnt * sqq - sq * sq;
                if (det > 0.0f) {
                    const float s2 = (cnt * sqw - sq * sw) / det;
                    if (s2 > 0.0f) {
                        s = s2;
                        m = (sw - s2 * sq) / cnt;
                        codes_for(v, s, m, q);
                    }
                }
                // Error with the values as they will be stored (f16 scale and min).
                const float sh = __half2float(__float2half_rn(s)), mh = __half2float(__float2half_rn(m));
                float err = 0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float d = static_cast<float>(q[i]) * sh + mh - v[i];
                    err += d * d;
                }
                err = warp_sum_all(err);
                if (err < best_err) {
                    best_err = err;
                    best_s = sh;
                    best_m = mh;
                }
            }
        }
    }
    uint32_t q[4];
    codes_for(v, best_s, best_m, q);
    // This lane holds k = 4 * lane + i: word w = lane / 2, positions p = 4 * (lane & 1) + i.
    // Position p of a word: even p -> nibble p / 2, odd p -> nibble 4 + p / 2.
    uint32_t part = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int p = 4 * (lane & 1) + i;
        part |= q[i] << (4 * ((p & 1) ? 4 + p / 2 : p / 2));
    }
    const uint32_t word = part | __shfl_xor_sync(0xffffffffu, part, 1);
    if ((lane & 1) == 0)
        reinterpret_cast<uint32_t *>(Wq + static_cast<size_t>(n) * (K / 2) + grp * 64)[lane / 2] = word;
    if (lane == 0) sm[static_cast<size_t>(n) * groups + grp] = __floats2half2_rn(best_s, best_m);
}

__global__ void dequantize_q8_kernel(const int8_t *Wq, const float *scales, int N, int K, half *out) {
    const int64_t total = static_cast<int64_t>(N) * K;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        out[i] = __float2half_rn(static_cast<float>(Wq[i]) * scales[i / K]);
}

// One thread per packed word (8 weights).
__global__ void dequantize_q4_kernel(const uint8_t *Wq, const half2 *sm, int N, int K, half *out) {
    const int64_t words = static_cast<int64_t>(N) * K / 8;
    const int groups = K / kInt4Group;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < words;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const uint32_t w = reinterpret_cast<const uint32_t *>(Wq)[i];
        const int64_t k0 = i * 8, n = k0 / K;
        const int k = static_cast<int>(k0 % K);
        const float2 s = __half22float2(sm[n * groups + k / kInt4Group]);
#pragma unroll
        for (int p = 0; p < 8; p++) {
            const uint32_t q = (w >> (4 * ((p & 1) ? 4 + p / 2 : p / 2))) & 15u;
            out[n * K + k + p] = __float2half_rn(static_cast<float>(q) * s.x + s.y);
        }
    }
}

template <int MT, Epilogue EP>
void launch_q(const half *A, int lda, const void *Wq, const half *scales, QuantType qt, const float *xsum, int M, int N,
              int K, void *C, int ldc, cudaStream_t s) {
    const int blocks = ceil_div(N, 16);
    if (qt == QuantType::int8)
        gemv_q8_kernel<MT, EP><<<blocks, QWARPS * 32, 0, s>>>(A, lda, static_cast<const int8_t *>(Wq),
                                                             reinterpret_cast<const float *>(scales), M, N, K, C, ldc);
    else
        gemv_q4_kernel<MT, EP><<<blocks, QWARPS * 32, 0, s>>>(A, lda, static_cast<const uint8_t *>(Wq),
                                                             reinterpret_cast<const half2 *>(scales), xsum, M, N, K, C, ldc);
}

template <Epilogue EP>
void dispatch_q(const half *A, int lda, const void *Wq, const half *scales, QuantType qt, const float *xsum, int M, int N,
                int K, void *C, int ldc, cudaStream_t s) {
    switch ((M + 7) / 8) {
        case 1: launch_q<1, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 2: launch_q<2, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 3: launch_q<3, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 4: launch_q<4, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 5: launch_q<5, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 6: launch_q<6, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 7: launch_q<7, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        case 8: launch_q<8, EP>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); return;
        default: fail("quantized gemv supports M <= 64, got {}", M);
    }
}

int grid_for(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return static_cast<int>(blocks < 65535 * 8 ? blocks : 65535 * 8);
}

}  // namespace

void gemm_quant(const half *A, int lda, const void *Wq, const half *scales, QuantType qt, int M, int N, int K, void *C,
                int ldc, Epilogue ep, const QuantScratch &scratch, cudaStream_t s) {
    if (M <= 0) return;
    EMBER_CHECK(qt != QuantType::none, "gemm_quant: weights are not quantized");
    EMBER_CHECK(K % kInt4Group == 0, "quantized weights need K % 128 == 0 (K = {})", K);
    if (M > 64) {
        // Prefill: expand to f16 once, then the tensor-core GEMM.
        EMBER_CHECK(scratch.dequant && scratch.dequant_elems >= static_cast<size_t>(N) * K, "gemm_quant: dequant scratch too small");
        dequantize_weights(Wq, scales, qt, N, K, scratch.dequant, s);
        gemm(A, lda, scratch.dequant, M, N, K, C, ldc, ep, s, GemmKernel::automatic, scratch.ws);
        return;
    }
    const float *xsum = nullptr;
    if (qt == QuantType::int4) {
        const int items = M * (K / kInt4Group);
        EMBER_CHECK(scratch.xsum && scratch.xsum_elems >= static_cast<size_t>(items), "gemm_quant: xsum scratch too small");
        group_sums_kernel<<<ceil_div(items, 8), 256, 0, s>>>(A, lda, M, K, scratch.xsum);
        xsum = scratch.xsum;
    }
    switch (ep) {
        case Epilogue::store_f16: dispatch_q<Epilogue::store_f16>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); break;
        case Epilogue::store_f32: dispatch_q<Epilogue::store_f32>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); break;
        case Epilogue::add_f32: dispatch_q<Epilogue::add_f32>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); break;
        case Epilogue::silu_mul: dispatch_q<Epilogue::silu_mul>(A, lda, Wq, scales, qt, xsum, M, N, K, C, ldc, s); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void quantize_weights(const half *W, int N, int K, QuantType qt, void *Wq, half *scales, cudaStream_t s) {
    EMBER_CHECK(K % kInt4Group == 0, "quantization needs K % 128 == 0 (K = {})", K);
    if (qt == QuantType::int8)
        quantize_q8_kernel<<<N, 256, 0, s>>>(W, K, static_cast<int8_t *>(Wq), reinterpret_cast<float *>(scales));
    else
        quantize_q4_kernel<<<ceil_div(N * (K / kInt4Group), 8), 256, 0, s>>>(W, N, K, static_cast<uint8_t *>(Wq),
                                                                             reinterpret_cast<half2 *>(scales));
    CUDA_CHECK(cudaGetLastError());
}

void dequantize_weights(const void *Wq, const half *scales, QuantType qt, int N, int K, half *out, cudaStream_t s) {
    if (qt == QuantType::int8)
        dequantize_q8_kernel<<<grid_for(static_cast<int64_t>(N) * K, 256), 256, 0, s>>>(
            static_cast<const int8_t *>(Wq), reinterpret_cast<const float *>(scales), N, K, out);
    else
        dequantize_q4_kernel<<<grid_for(static_cast<int64_t>(N) * K / 8, 256), 256, 0, s>>>(
            static_cast<const uint8_t *>(Wq), reinterpret_cast<const half2 *>(scales), N, K, out);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ember::cuda

namespace ember::cuda {

namespace {
// Embedding rows straight from a quantized (tied) table.
__global__ void embed_q_kernel(const int32_t *tokens, const void *Wq, const half *scales, QuantType qt, int hidden, float *out) {
    const int t = blockIdx.x, tok = tokens[t];
    float *o = out + static_cast<size_t>(t) * hidden;
    if (qt == QuantType::int8) {
        const int8_t *row = static_cast<const int8_t *>(Wq) + static_cast<size_t>(tok) * hidden;
        const float s = reinterpret_cast<const float *>(scales)[tok];
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) o[i] = static_cast<float>(row[i]) * s;
    } else {
        const int groups = hidden / kInt4Group;
        const uint32_t *row = reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(Wq) + static_cast<size_t>(tok) * (hidden / 2));
        const half2 *sm = reinterpret_cast<const half2 *>(scales) + static_cast<size_t>(tok) * groups;
        for (int w = threadIdx.x; w < hidden / 8; w += blockDim.x) {
            const uint32_t word = row[w];
            const float2 s = __half22float2(sm[(w * 8) / kInt4Group]);
#pragma unroll
            for (int p = 0; p < 8; p++) {
                const uint32_t q = (word >> (4 * ((p & 1) ? 4 + p / 2 : p / 2))) & 15u;
                o[w * 8 + p] = static_cast<float>(q) * s.x + s.y;
            }
        }
    }
}
}  // namespace

void embed_tokens_quant(const int32_t *tokens, int T, const void *Wq, const half *scales, QuantType qt, int hidden, float *out,
                        cudaStream_t s) {
    if (T <= 0) return;
    embed_q_kernel<<<T, 256, 0, s>>>(tokens, Wq, scales, qt, hidden, out);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ember::cuda
