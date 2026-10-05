// Matrix multiplication C[M, N] = A[M, K] * W[N, K]^T, the bulk of the work.
//
// Two regimes, two kernels:
//  - M <= 8 (decode): the time is spent reading W, once. gemv_kernel streams
//    each weight row with 16-byte loads and reuses it for all M tokens.
//  - M > 8 (prefill, large batches): tensor cores. A tile of A and of W is
//    staged in shared memory, and each warp computes a sub-tile with
//    mma.sync m16n8k8 (the Turing instruction; also valid on later GPUs).
//
// Both end in a fused epilogue: write f16, write f32, add into the f32
// residual stream, or compute silu(gate) * up for the MLP while the values are
// still in registers.
#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/kernels.cuh"

#include <algorithm>

namespace ember::cuda {

namespace {

constexpr float kHalfMax = 65504.0f;

__device__ __forceinline__ half to_half_sat(float v) { return __float2half_rn(fminf(fmaxf(v, -kHalfMax), kHalfMax)); }

__device__ __forceinline__ float silu(float g) { return g / (1.0f + __expf(-g)); }

// Writes C[r, c] and C[r, c + 1] (c is even) according to the epilogue.
template <Epilogue EP>
__device__ __forceinline__ void store_pair(void *C, int ldc, int M, int N, int r, int c, float v0, float v1) {
    if (r >= M || c >= N) return;
    const bool both = c + 1 < N;
    if constexpr (EP == Epilogue::store_f16) {
        half *p = static_cast<half *>(C) + static_cast<size_t>(r) * ldc + c;
        if (both) *reinterpret_cast<half2 *>(p) = __halves2half2(to_half_sat(v0), to_half_sat(v1));
        else p[0] = to_half_sat(v0);
    } else if constexpr (EP == Epilogue::store_f32) {
        float *p = static_cast<float *>(C) + static_cast<size_t>(r) * ldc + c;
        if (both) *reinterpret_cast<float2 *>(p) = make_float2(v0, v1);
        else p[0] = v0;
    } else if constexpr (EP == Epilogue::add_f32) {
        float *p = static_cast<float *>(C) + static_cast<size_t>(r) * ldc + c;
        if (both) {
            float2 o = *reinterpret_cast<float2 *>(p);
            *reinterpret_cast<float2 *>(p) = make_float2(o.x + v0, o.y + v1);
        } else {
            p[0] += v0;
        }
    } else {  // silu_mul: (c, c + 1) = (gate_j, up_j) with j = c / 2
        half *p = static_cast<half *>(C) + static_cast<size_t>(r) * ldc + c / 2;
        *p = to_half_sat(silu(v0) * v1);
    }
}

// ---- tensor-core GEMM ----------------------------------------------------------

constexpr int BK = 32;       // K per shared-memory tile (two mma k-steps of 16 via ldmatrix x4)
constexpr int SPAD = 8;      // row padding: 40 halves = 80 bytes, so ldmatrix rows hit distinct banks
constexpr int LDS = BK + SPAD;

// With gridDim.z > 1 the K dimension is split ("split-K"): block z covers
// [z * k_split, (z + 1) * k_split). The residual-stream epilogue adds its
// partial result atomically; the others write fp32 partials to `partial`
// ([splits][M][N]) for finalize_split_kernel to sum and finish.
template <int BM, int BN, int WARPS_M, int WARPS_N, Epilogue EP>
__global__ void __launch_bounds__(WARPS_M *WARPS_N * 32)
    gemm_tc_kernel(const half *__restrict__ A, int lda, const half *__restrict__ W, int M, int N, int K,
                   void *__restrict__ C, int ldc, int k_split, float *__restrict__ partial) {
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MI = WM / 16, NI = WN / 8;
    static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tile must be a multiple of 16x16");
    constexpr int A_CHUNKS = BM * BK / 8, B_CHUNKS = BN * BK / 8;  // 16-byte chunks per tile
    constexpr int A_ITERS = (A_CHUNKS + THREADS - 1) / THREADS, B_ITERS = (B_CHUNKS + THREADS - 1) / THREADS;

    __shared__ __align__(16) half sA[2][BM * LDS];
    __shared__ __align__(16) half sB[2][BN * LDS];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const int k_begin = blockIdx.z * k_split, k_end = min(K, k_begin + k_split);

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;

    uint4 ra[A_ITERS], rb[B_ITERS];
    const uint4 zero = make_uint4(0, 0, 0, 0);

    auto load_global = [&](int k0) {
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int c = tid + i * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                int row = c >> 2, col = (c & 3) * 8;
                int gr = bm + row, gk = k0 + col;
                ra[i] = (gr < M && gk < k_end) ? *reinterpret_cast<const uint4 *>(A + static_cast<size_t>(gr) * lda + gk) : zero;
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; i++) {
            int c = tid + i * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                int row = c >> 2, col = (c & 3) * 8;
                int gn = bn + row, gk = k0 + col;
                rb[i] = (gn < N && gk < k_end) ? __ldg(reinterpret_cast<const uint4 *>(W + static_cast<size_t>(gn) * K + gk)) : zero;
            }
        }
    };
    auto store_shared = [&](int buf) {
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int c = tid + i * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS)
                *reinterpret_cast<uint4 *>(&sA[buf][(c >> 2) * LDS + (c & 3) * 8]) = ra[i];
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; i++) {
            int c = tid + i * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS)
                *reinterpret_cast<uint4 *>(&sB[buf][(c >> 2) * LDS + (c & 3) * 8]) = rb[i];
        }
    };

    const int ktiles = (k_end - k_begin + BK - 1) / BK;
    load_global(k_begin);
    store_shared(0);
    __syncthreads();

    for (int kt = 0; kt < ktiles; kt++) {
        const int cur = kt & 1;
        // Fetch the next tile into registers while this one is being multiplied.
        if (kt + 1 < ktiles) load_global(k_begin + (kt + 1) * BK);

#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t af[MI][4];
            uint32_t bf[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                int row = wm * WM + mi * 16 + (lane & 15);
                int col = ks * 16 + (lane >> 4) * 8;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], &sA[cur][row * LDS + col]);
            }
#pragma unroll
            for (int nj = 0; nj < NI / 2; nj++) {
                int row = wn * WN + nj * 16 + (lane >> 4) * 8 + (lane & 7);
                int col = ks * 16 + ((lane >> 3) & 1) * 8;
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, &sB[cur][row * LDS + col]);
                bf[2 * nj][0] = b0;
                bf[2 * nj][1] = b1;
                bf[2 * nj + 1][0] = b2;
                bf[2 * nj + 1][1] = b3;
            }
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int ni = 0; ni < NI; ni++) {
                    mma_16x8x8(acc[mi][ni], af[mi][0], af[mi][1], bf[ni][0]);
                    mma_16x8x8(acc[mi][ni], af[mi][2], af[mi][3], bf[ni][1]);
                }
        }

        if (kt + 1 < ktiles) store_shared(cur ^ 1);
        __syncthreads();
    }

    // Accumulator layout of m16n8: c0, c1 at (row = lane / 4, col = 2 * (lane % 4) + {0, 1}); c2, c3 eight rows down.
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
            int r = bm + wm * WM + mi * 16 + (lane >> 2);
            int c = bn + wn * WN + ni * 8 + (lane & 3) * 2;
            if (gridDim.z == 1) {
                store_pair<EP>(C, ldc, M, N, r, c, acc[mi][ni][0], acc[mi][ni][1]);
                store_pair<EP>(C, ldc, M, N, r + 8, c, acc[mi][ni][2], acc[mi][ni][3]);
            } else {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int rr = r + 8 * h;
                    if (rr >= M || c >= N) continue;
                    const float v0 = acc[mi][ni][2 * h], v1 = acc[mi][ni][2 * h + 1];
                    if constexpr (EP == Epilogue::add_f32) {
                        float *p = static_cast<float *>(C) + static_cast<size_t>(rr) * ldc + c;
                        atomicAdd(p, v0);
                        if (c + 1 < N) atomicAdd(p + 1, v1);
                    } else {
                        float *p = partial + (static_cast<size_t>(blockIdx.z) * M + rr) * N + c;
                        p[0] = v0;
                        if (c + 1 < N) p[1] = v1;
                    }
                }
            }
        }
}

// Sums the split-K partials and applies the epilogue; one thread per column pair.
template <Epilogue EP>
__global__ void finalize_split_kernel(const float *__restrict__ partial, int splits, int M, int N, void *__restrict__ C,
                                      int ldc) {
    const int pairs = (N + 1) / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * pairs) return;
    const int r = i / pairs, c = (i % pairs) * 2;
    float v0 = 0.0f, v1 = 0.0f;
    for (int z = 0; z < splits; z++) {
        const float *p = partial + (static_cast<size_t>(z) * M + r) * N + c;
        v0 += p[0];
        if (c + 1 < N) v1 += p[1];
    }
    store_pair<EP>(C, ldc, M, N, r, c, v0, v1);
}

// ---- GEMV for small M --------------------------------------------------------------

__device__ __forceinline__ float dot8(const uint4 &a, const uint4 &b) {
    const half2 *ha = reinterpret_cast<const half2 *>(&a);
    const half2 *hb = reinterpret_cast<const half2 *>(&b);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 fa = __half22float2(ha[i]), fb = __half22float2(hb[i]);
        s = fmaf(fa.x, fb.x, s);
        s = fmaf(fa.y, fb.y, s);
    }
    return s;
}

// Streams weights past the L1 cache: each byte is used once per step.
__device__ __forceinline__ uint4 load_stream(const half *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

constexpr int GEMV_WARPS = 8;

// Each warp computes R consecutive output columns for all M tokens.
template <int M, int R, Epilogue EP>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
    gemv_kernel(const half *__restrict__ A, int lda, const half *__restrict__ W, int N, int K, void *__restrict__ C,
                int ldc) {
    const int lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5)) * R;
    if (n0 >= N) return;

    float acc[M][R];
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[m][r] = 0.0f;

    const half *wrow[R];
#pragma unroll
    for (int r = 0; r < R; r++) wrow[r] = W + static_cast<size_t>(min(n0 + r, N - 1)) * K;

#pragma unroll 4
    for (int k = lane * 8; k < K; k += 32 * 8) {
        uint4 w[R];
#pragma unroll
        for (int r = 0; r < R; r++) w[r] = load_stream(wrow[r] + k);
#pragma unroll
        for (int m = 0; m < M; m++) {
            uint4 x = *reinterpret_cast<const uint4 *>(A + static_cast<size_t>(m) * lda + k);
#pragma unroll
            for (int r = 0; r < R; r++) acc[m][r] += dot8(w[r], x);
        }
    }
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[m][r] = warp_sum(acc[m][r]);

    // Every lane now holds every sum; lanes split the stores.
    if constexpr (EP == Epilogue::silu_mul) {
        static_assert(R % 2 == 0, "silu_mul needs gate/up pairs");
#pragma unroll
        for (int m = 0; m < M; m++)
#pragma unroll
            for (int r = 0; r < R; r += 2)
                if (lane == m * (R / 2) + r / 2 && n0 + r + 1 < N)
                    store_pair<EP>(C, ldc, M, N, m, n0 + r, acc[m][r], acc[m][r + 1]);
    } else {
#pragma unroll
        for (int m = 0; m < M; m++)
#pragma unroll
            for (int r = 0; r < R; r += 2)
                if (lane == (m * R + r) / 2 % 32) store_pair<EP>(C, ldc, M, N, m, n0 + r, acc[m][r], acc[m][r + 1]);
    }
}


// ---- tensor-core GEMV for decode-sized batches (M <= 64) ----------------------------
//
// In this regime the time is spent streaming W, so the kernel is built around
// the loads: no shared-memory staging, no __syncthreads in the main loop.
//
// The trick: a dot product does not care about the order of k, so A and B may
// use any common permutation of k. Thread (g = lane / 4, t = lane % 4) loads
// 16 contiguous bytes, W[row][kb + 8t .. kb + 8t + 7], and uses its half2 j as
// the fragment for "virtual" k = 2t, 2t + 1 of mma step j. Loading the
// activations with the same pattern lines both operands up, so 16-byte
// coalesced loads feed mma.sync directly.
//
// Roles are swapped relative to the usual GEMM: the mma's M side is 16 rows of
// W (output features) and its N side is 8 tokens, so the accumulators hold
// C^T. A block of 4 warps covers 16 output rows; the warps split K four ways
// and sum their partials in shared memory at the end.
constexpr int GEMV_TC_WARPS = 4;

__device__ __forceinline__ uint4 load_x16(const half *p) { return *reinterpret_cast<const uint4 *>(p); }

template <int MT, Epilogue EP>  // MT = token tiles of 8 (M <= 8 * MT)
__global__ void __launch_bounds__(GEMV_TC_WARPS * 32)
    gemv_tc_kernel(const half *__restrict__ A, int lda, const half *__restrict__ W, int M, int N, int K,
                   void *__restrict__ C, int ldc) {
    __shared__ float red[GEMV_TC_WARPS - 1][MT][4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    const int row_a = min(n0 + g, N - 1), row_b = min(n0 + g + 8, N - 1);
    const half *wa = W + static_cast<size_t>(row_a) * K + 8 * t;
    const half *wb = W + static_cast<size_t>(row_b) * K + 8 * t;

    // Token rows this thread loads for each tile (clamped; rows >= M are zeroed).
    const half *xr[MT];
    bool xok[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int tok = mt * 8 + g;
        xok[mt] = tok < M;
        xr[mt] = A + static_cast<size_t>(min(tok, M - 1)) * lda + 8 * t;
    }

    float acc[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;

    const int chunks = K / 32;  // 32 k per chunk: 4 threads x 8 halves
#pragma unroll 2
    for (int c = warp; c < chunks; c += GEMV_TC_WARPS) {
        const int kb = c * 32;
        const uint4 a_lo = load_stream(wa + kb), a_hi = load_stream(wb + kb);
        uint4 x[MT];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) x[mt] = xok[mt] ? load_x16(xr[mt] + kb) : make_uint4(0, 0, 0, 0);
        const uint32_t *al = reinterpret_cast<const uint32_t *>(&a_lo);
        const uint32_t *ah = reinterpret_cast<const uint32_t *>(&a_hi);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const uint32_t *xb = reinterpret_cast<const uint32_t *>(&x[mt]);
#pragma unroll
            for (int j = 0; j < 4; j++) mma_16x8x8(acc[mt], al[j], ah[j], xb[j]);
        }
    }

    // Sum the four warps' partial results.
    if (warp > 0) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int e = 0; e < 4; e++) red[warp - 1][mt][e][lane] = acc[mt][e];
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int e = 0; e < 4; e++)
#pragma unroll
            for (int w = 0; w < GEMV_TC_WARPS - 1; w++) acc[mt][e] += red[w][mt][e][lane];

    // acc[mt] holds C^T: (row n0 + g, tokens 2t, 2t + 1) and (row n0 + g + 8, same tokens).
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int tok = mt * 8 + 2 * t + (e & 1);
            const int n = n0 + g + (e >> 1) * 8;
            const float v = acc[mt][e];
            if constexpr (EP == Epilogue::silu_mul) {
                // gate rows are even, up rows odd: the partner row n + 1 lives in lane + 4.
                const float up = __shfl_down_sync(0xffffffffu, v, 4);
                if ((g & 1) == 0 && tok < M && n + 1 < N)
                    static_cast<half *>(C)[static_cast<size_t>(tok) * ldc + n / 2] = to_half_sat(silu(v) * up);
            } else {
                if (tok >= M || n >= N) continue;
                if constexpr (EP == Epilogue::store_f16)
                    static_cast<half *>(C)[static_cast<size_t>(tok) * ldc + n] = to_half_sat(v);
                else if constexpr (EP == Epilogue::store_f32)
                    static_cast<float *>(C)[static_cast<size_t>(tok) * ldc + n] = v;
                else
                    static_cast<float *>(C)[static_cast<size_t>(tok) * ldc + n] += v;
            }
        }
    }
}

template <int MT, Epilogue EP>
void launch_gemv_tc(const half *A, int lda, const half *W, int M, int N, int K, void *C, int ldc, cudaStream_t s) {
    gemv_tc_kernel<MT, EP><<<ceil_div(N, 16), GEMV_TC_WARPS * 32, 0, s>>>(A, lda, W, M, N, K, C, ldc);
}

template <Epilogue EP>
void dispatch_gemv_tc(const half *A, int lda, const half *W, int M, int N, int K, void *C, int ldc, cudaStream_t s) {
    switch ((M + 7) / 8) {
        case 1: launch_gemv_tc<1, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 2: launch_gemv_tc<2, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 3: launch_gemv_tc<3, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 4: launch_gemv_tc<4, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 5: launch_gemv_tc<5, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 6: launch_gemv_tc<6, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 7: launch_gemv_tc<7, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        case 8: launch_gemv_tc<8, EP>(A, lda, W, M, N, K, C, ldc, s); return;
        default: fail("gemv_tc supports M <= 64, got {}", M);
    }
}

template <int BM, int BN, int WM_, int WN_, Epilogue EP>
void launch_tc(const half *A, int lda, const half *W, int M, int N, int K, void *C, int ldc, cudaStream_t s,
               const GemmWorkspace *ws) {
    const int blocks = ceil_div(N, BN) * ceil_div(M, BM);
    // Small grids leave SMs idle and starve the memory system: split K until
    // there are a few blocks per SM (each split keeps at least 4 k-tiles).
    int splits = 1;
    if (ws && ws->partial) {
        const int target = 4 * sm_count();
        splits = std::min({8, std::max(1, target / std::max(blocks, 1)), K / (4 * BK)});
        if (EP != Epilogue::add_f32)
            while (splits > 1 && static_cast<size_t>(splits) * M * N > ws->partial_elems) splits--;
        splits = std::max(splits, 1);
    }
    const int k_split = ceil_div(ceil_div(K, splits), BK) * BK;
    splits = ceil_div(K, k_split);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM), splits);
    gemm_tc_kernel<BM, BN, WM_, WN_, EP><<<grid, WM_ * WN_ * 32, 0, s>>>(A, lda, W, M, N, K, C, ldc, k_split,
                                                                        splits > 1 ? ws->partial : nullptr);
    if (splits > 1 && EP != Epilogue::add_f32) {
        const int work = M * ((N + 1) / 2);
        finalize_split_kernel<EP><<<ceil_div(work, 256), 256, 0, s>>>(ws->partial, splits, M, N, C, ldc);
    }
}

template <int M, Epilogue EP>
void launch_gemv(const half *A, int lda, const half *W, int N, int K, void *C, int ldc, cudaStream_t s) {
    constexpr int R = M <= 2 ? 4 : 2;
    int warps = ceil_div(N, R);
    gemv_kernel<M, R, EP><<<ceil_div(warps, GEMV_WARPS), GEMV_WARPS * 32, 0, s>>>(A, lda, W, N, K, C, ldc);
}

template <Epilogue EP>
void dispatch(const half *A, int lda, const half *W, int M, int N, int K, void *C, int ldc, cudaStream_t s,
              GemmKernel kernel, const GemmWorkspace *ws) {
    if (kernel == GemmKernel::automatic) {
        if (M <= 64 && K % 32 == 0) kernel = GemmKernel::gemv_tc;
        else if (M <= 8) kernel = GemmKernel::gemv;
        else if (M <= 16) kernel = GemmKernel::tc16;
        else if (M <= 32) kernel = GemmKernel::tc32;
        else if (M <= 64) kernel = GemmKernel::tc64;
        else kernel = GemmKernel::tc128;
    }
    switch (kernel) {
        case GemmKernel::gemv:
            switch (M) {
                case 1: launch_gemv<1, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 2: launch_gemv<2, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 3: launch_gemv<3, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 4: launch_gemv<4, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 5: launch_gemv<5, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 6: launch_gemv<6, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 7: launch_gemv<7, EP>(A, lda, W, N, K, C, ldc, s); return;
                case 8: launch_gemv<8, EP>(A, lda, W, N, K, C, ldc, s); return;
                default: fail("gemv supports M <= 8, got {}", M);
            }
        case GemmKernel::gemv_tc:
            EMBER_CHECK(K % 32 == 0 && M <= 64, "gemv_tc needs K % 32 == 0 and M <= 64");
            dispatch_gemv_tc<EP>(A, lda, W, M, N, K, C, ldc, s);
            return;
        case GemmKernel::tc16: launch_tc<16, 64, 1, 4, EP>(A, lda, W, M, N, K, C, ldc, s, ws); return;
        case GemmKernel::tc32: launch_tc<32, 128, 2, 4, EP>(A, lda, W, M, N, K, C, ldc, s, ws); return;
        case GemmKernel::tc64: launch_tc<64, 128, 2, 4, EP>(A, lda, W, M, N, K, C, ldc, s, ws); return;
        case GemmKernel::tc128: launch_tc<128, 128, 2, 4, EP>(A, lda, W, M, N, K, C, ldc, s, ws); return;
        default: fail("unknown GEMM kernel");
    }
}

}  // namespace

void gemm(const half *A, int lda, const half *W, int M, int N, int K, void *C, int ldc, Epilogue ep, cudaStream_t s,
          GemmKernel kernel, const GemmWorkspace *ws) {
    if (M <= 0 || N <= 0) return;
    EMBER_CHECK(K % 8 == 0 && lda % 8 == 0, "gemm: K and lda must be multiples of 8 (K={}, lda={})", K, lda);
    EMBER_CHECK(ep != Epilogue::silu_mul || N % 2 == 0, "gemm: silu_mul needs an even N");
    switch (ep) {
        case Epilogue::store_f16: dispatch<Epilogue::store_f16>(A, lda, W, M, N, K, C, ldc, s, kernel, ws); break;
        case Epilogue::store_f32: dispatch<Epilogue::store_f32>(A, lda, W, M, N, K, C, ldc, s, kernel, ws); break;
        case Epilogue::add_f32: dispatch<Epilogue::add_f32>(A, lda, W, M, N, K, C, ldc, s, kernel, ws); break;
        case Epilogue::silu_mul: dispatch<Epilogue::silu_mul>(A, lda, W, M, N, K, C, ldc, s, kernel, ws); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ember::cuda
