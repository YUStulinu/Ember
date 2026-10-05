// Attention over the paged KV cache.
//
// paged_attention: one block per (query token, kv head, context split). It
// serves decode (one query per sequence) and any token whose context must be
// read from the cache. The G query heads sharing a kv head (GQA) are handled
// together, so each key and value is read once for all of them.
//
// Within a block, each warp walks whole pages (16 keys, contiguous in memory):
//   scores: two lanes per key, each covering half of head_dim, then one shuffle;
//   values: each lane owns head_dim / 32 output dimensions.
// Softmax is online (running max and sum), in base 2 with the scale folded
// into q. Long contexts are split across blocks and merged by a second kernel.
#include "backend/backend.hpp"
#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/kernels.cuh"

namespace ember::cuda {

namespace {

constexpr int kWarps = 4;
constexpr float kLog2e = 1.4426950408889634f;

template <int D>
__device__ __forceinline__ int qidx(int d) {
    return d + (d >= D / 2 ? 4 : 0);  // pad the second half: the two lane halves hit different banks
}

template <int D, int G>
__global__ void __launch_bounds__(kWarps * 32) paged_attention_kernel(AttentionArgs a) {
    constexpr int E = D / 32;      // value dims per lane
    constexpr int HALF = D / 2;    // key dims per lane (two lanes per key)
    constexpr int QS = D + 4;

    const int t = a.token_index ? a.token_index[blockIdx.x] : blockIdx.x;
    const int kvh = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ctx = a.positions[t] + 1;
    const int begin = split * a.split_len;
    const int end = min(ctx, begin + a.split_len);

    __shared__ float q_s[G][QS];
    __shared__ float red_m[kWarps][G], red_l[kWarps][G];
    __shared__ float red_acc[kWarps][G][D];

    for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
        int g = i / D, d = i % D;
        float v = __half2float(a.q[(static_cast<size_t>(t) * a.n_heads + kvh * G + g) * D + d]);
        q_s[g][qidx<D>(d)] = v * a.scale * kLog2e;
    }
    __syncthreads();

    float m[G], l[G], acc[G][E];
#pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int e = 0; e < E; e++) acc[g][e] = 0.0f;
    }

    const int seq = a.token_seq[t];
    const int32_t *table = a.block_tables + static_cast<size_t>(seq) * a.max_blocks;
    const int first_page = begin / kBlockSize, last_page = (end - 1) / kBlockSize;
    const int key = lane >> 1, part = lane & 1;

    if (begin < end) {
        for (int p = first_page + warp; p <= last_page; p += kWarps) {
            const size_t base = (static_cast<size_t>(table[p]) * a.n_kv + kvh) * kBlockSize * D;
            const int pos = p * kBlockSize + key;
            const bool valid = pos >= begin && pos < end;

            // Scores for this lane's key, over its half of the dimensions.
            float s[G];
#pragma unroll
            for (int g = 0; g < G; g++) s[g] = 0.0f;
            const half *krow = a.k_cache + base + key * D + part * HALF;
#pragma unroll
            for (int c = 0; c < HALF / 8; c++) {
                uint4 raw = *reinterpret_cast<const uint4 *>(krow + c * 8);
                const half2 *h2 = reinterpret_cast<const half2 *>(&raw);
                float kf[8];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 f = __half22float2(h2[i]);
                    kf[2 * i] = f.x;
                    kf[2 * i + 1] = f.y;
                }
#pragma unroll
                for (int g = 0; g < G; g++) {
                    const float *qq = &q_s[g][qidx<D>(part * HALF + c * 8)];
#pragma unroll
                    for (int i = 0; i < 8; i++) s[g] = fmaf(qq[i], kf[i], s[g]);
                }
            }

            float pj[G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                s[g] += __shfl_xor_sync(0xffffffffu, s[g], 1);
                if (!valid) s[g] = -INFINITY;
                float page_max = warp_max(s[g]);
                float m_new = fmaxf(m[g], page_max);
                float corr = exp2f(m[g] - m_new);  // m[g] = -inf on the first page: corr = 0
                pj[g] = valid ? exp2f(s[g] - m_new) : 0.0f;
                l[g] = l[g] * corr + warp_sum(part == 0 ? pj[g] : 0.0f);
#pragma unroll
                for (int e = 0; e < E; e++) acc[g][e] *= corr;
                m[g] = m_new;
            }

            // Values: lane owns dims [lane * E, lane * E + E). Only valid keys: slots past
            // the end were never written and may hold anything (0 * NaN is NaN).
            const half *vbase = a.v_cache + base + lane * E;
            const int j_lo = max(0, begin - p * kBlockSize), j_hi = min(kBlockSize, end - p * kBlockSize);
#pragma unroll 4
            for (int j = j_lo; j < j_hi; j++) {
                float vf[E];
                if constexpr (E == 4) {
                    uint2 raw = *reinterpret_cast<const uint2 *>(vbase + j * D);
                    const half2 *h2 = reinterpret_cast<const half2 *>(&raw);
                    float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]);
                    vf[0] = f0.x;
                    vf[1] = f0.y;
                    vf[2] = f1.x;
                    vf[3] = f1.y;
                } else {
#pragma unroll
                    for (int e = 0; e < E; e++) vf[e] = __half2float(vbase[j * D + e]);
                }
#pragma unroll
                for (int g = 0; g < G; g++) {
                    float w = __shfl_sync(0xffffffffu, pj[g], j * 2);
#pragma unroll
                    for (int e = 0; e < E; e++) acc[g][e] = fmaf(w, vf[e], acc[g][e]);
                }
            }
        }
    }

    // Merge the warps.
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (lane == 0) {
            red_m[warp][g] = m[g];
            red_l[warp][g] = l[g];
        }
#pragma unroll
        for (int e = 0; e < E; e++) red_acc[warp][g][lane * E + e] = acc[g][e];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
        const int g = i / D, d = i % D;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < kWarps; w++) M = fmaxf(M, red_m[w][g]);
        float L = 0.0f, O = 0.0f;
        if (M != -INFINITY) {
#pragma unroll
            for (int w = 0; w < kWarps; w++) {
                float f = exp2f(red_m[w][g] - M);
                L += red_l[w][g] * f;
                O += red_acc[w][g][d] * f;
            }
        }
        const int head = kvh * G + g;
        if (a.num_splits == 1) {
            a.out[(static_cast<size_t>(t) * a.n_heads + head) * D + d] = __float2half_rn(L > 0.0f ? O / L : 0.0f);
        } else {
            const size_t slot = (static_cast<size_t>(t) * a.n_heads + head) * a.num_splits + split;
            a.part_acc[slot * D + d] = O;
            if (d == 0) {
                a.part_ml[slot * 2] = M;
                a.part_ml[slot * 2 + 1] = L;
            }
        }
    }
}

// Merges the splits of one (token, head): out = sum_s O_s 2^(M_s - M) / sum_s L_s 2^(M_s - M).
__global__ void combine_splits_kernel(AttentionArgs a) {
    const int t = a.token_index ? a.token_index[blockIdx.x] : blockIdx.x;
    const int head = blockIdx.y, D = a.head_dim;
    const size_t base = (static_cast<size_t>(t) * a.n_heads + head) * a.num_splits;
    float M = -INFINITY;
    for (int s = 0; s < a.num_splits; s++) M = fmaxf(M, a.part_ml[(base + s) * 2]);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float L = 0.0f, O = 0.0f;
        if (M != -INFINITY) {
            for (int s = 0; s < a.num_splits; s++) {
                float ms = a.part_ml[(base + s) * 2];
                if (ms == -INFINITY) continue;
                float f = exp2f(ms - M);
                L += a.part_ml[(base + s) * 2 + 1] * f;
                O += a.part_acc[(base + s) * D + d] * f;
            }
        }
        a.out[(static_cast<size_t>(t) * a.n_heads + head) * D + d] = __float2half_rn(L > 0.0f ? O / L : 0.0f);
    }
}


// ---- prefill: FlashAttention-2 on tensor cores --------------------------------------
//
// Block = 4 warps = 64 query rows (16 per warp) of one sequence, one head.
// Q stays in registers as mma A-fragments. For each tile of 32 keys:
//   S = Q K^T          (16 x 32 per warp; K tile in shared memory, ldmatrix)
//   online softmax     (row max/sum across the 4 lanes sharing a row)
//   O += P V           (P reused straight from the S accumulators as A-fragments,
//                       V read with ldmatrix.trans as the B operand)
constexpr int kKeyTile = 32;

template <int D>
__global__ void __launch_bounds__(128) prefill_attention_kernel(PrefillArgs a) {
    constexpr int SROW = D + 8;            // padded smem row (halves): conflict-free ldmatrix
    constexpr int KS = D / 8;              // k8 steps over the head dimension
    constexpr int NT = D / 8;              // n8 tiles of the output
    __shared__ __align__(16) half sK[kKeyTile * SROW];
    __shared__ __align__(16) half sV[kKeyTile * SROW];

    const int tile = blockIdx.x, head = blockIdx.y, kvh = head / (a.n_heads / a.n_kv);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int t0 = a.tile_start[tile], t_end = a.tile_end[tile], seq = a.tile_seq[tile];
    const int32_t *table = a.block_tables + static_cast<size_t>(seq) * a.max_blocks;
    const int kv_len = a.positions[t_end - 1] + 1;

    // This thread's two query rows and their positions (-1: row beyond the tile).
    const int r0 = t0 + warp * 16 + g, r1 = r0 + 8;
    const int pos0 = r0 < t_end ? a.positions[r0] : -1, pos1 = r1 < t_end ? a.positions[r1] : -1;
    const bool warp_active = t0 + warp * 16 < t_end;

    // Q fragments: a0 = Q[r0][k..k+1], a1 = Q[r1][k..k+1] with k = 8 * step + 2 * tq.
    uint32_t qf[KS][2];
    {
        const half *q0 = a.q + (static_cast<size_t>(min(r0, t_end - 1)) * a.n_heads + head) * D + 2 * tq;
        const half *q1 = a.q + (static_cast<size_t>(min(r1, t_end - 1)) * a.n_heads + head) * D + 2 * tq;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            qf[ks][0] = *reinterpret_cast<const uint32_t *>(q0 + ks * 8);
            qf[ks][1] = *reinterpret_cast<const uint32_t *>(q1 + ks * 8);
        }
    }

    float o[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float sl2 = a.scale * kLog2e;

    for (int kv0 = 0; kv0 < kv_len; kv0 += kKeyTile) {
        // Load 32 keys and values (two pages of the cache) into shared memory;
        // slots past the context are zero (they may never have been written).
        constexpr int CHUNKS = kKeyTile * D / 8;  // 16-byte pieces per tensor
        for (int c = threadIdx.x; c < CHUNKS; c += blockDim.x) {
            const int key = c / (D / 8), col = (c % (D / 8)) * 8;
            const int pos = kv0 + key;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (pos < kv_len) {
                const size_t off =
                    ((static_cast<size_t>(table[pos / kBlockSize]) * a.n_kv + kvh) * kBlockSize + pos % kBlockSize) * D + col;
                kv = *reinterpret_cast<const uint4 *>(a.k_cache + off);
                vv = *reinterpret_cast<const uint4 *>(a.v_cache + off);
            }
            *reinterpret_cast<uint4 *>(&sK[key * SROW + col]) = kv;
            *reinterpret_cast<uint4 *>(&sV[key * SROW + col]) = vv;
        }
        __syncthreads();

        if (warp_active) {
            // S = Q K^T for 32 keys: 4 n8 tiles.
            float sc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
            for (int k16 = 0; k16 < D / 16; k16++) {
#pragma unroll
                for (int nt2 = 0; nt2 < 2; nt2++) {
                    uint32_t b0, b1, b2, b3;
                    const int row = nt2 * 16 + (lane >> 4) * 8 + (lane & 7);
                    const int col = k16 * 16 + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(b0, b1, b2, b3, &sK[row * SROW + col]);
                    mma_16x8x8(sc[2 * nt2], qf[2 * k16][0], qf[2 * k16][1], b0);
                    mma_16x8x8(sc[2 * nt2], qf[2 * k16 + 1][0], qf[2 * k16 + 1][1], b1);
                    mma_16x8x8(sc[2 * nt2 + 1], qf[2 * k16][0], qf[2 * k16][1], b2);
                    mma_16x8x8(sc[2 * nt2 + 1], qf[2 * k16 + 1][0], qf[2 * k16 + 1][1], b3);
                }
            }
            // Scale, causal mask, row maxima.
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int key = kv0 + nt * 8 + 2 * tq + e;
                    float v0 = sc[nt][e] * sl2, v1 = sc[nt][2 + e] * sl2;
                    if (key > pos0 || key >= kv_len) v0 = -INFINITY;
                    if (key > pos1 || key >= kv_len) v1 = -INFINITY;
                    sc[nt][e] = v0;
                    sc[nt][2 + e] = v1;
                    mx0 = fmaxf(mx0, v0);
                    mx1 = fmaxf(mx1, v1);
                }
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
            // A row with nothing visible yet keeps m = -inf; use 0 as the reference so exp2 stays finite.
            const float ref0 = n0 == -INFINITY ? 0.0f : n0, ref1 = n1 == -INFINITY ? 0.0f : n1;
            const float c0 = exp2f(m0 - ref0), c1 = exp2f(m1 - ref1);
            float s0 = 0.0f, s1 = 0.0f;
            uint32_t pa[4][2];  // P as A-fragments, one per k8 step (= n8 tile of S)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = exp2f(sc[nt][0] - ref0), p01 = exp2f(sc[nt][1] - ref0);
                float p10 = exp2f(sc[nt][2] - ref1), p11 = exp2f(sc[nt][3] - ref1);
                s0 += p00 + p01;
                s1 += p10 + p11;
                pa[nt][0] = pack_half2(p00, p01);
                pa[nt][1] = pack_half2(p10, p11);
            }
            s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
            l0 = l0 * c0 + s0;
            l1 = l1 * c1 + s1;
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                o[nt][0] *= c0;
                o[nt][1] *= c0;
                o[nt][2] *= c1;
                o[nt][3] *= c1;
            }
            // O += P V: B-fragments of V via ldmatrix.trans (V is [key][dim] in shared memory).
#pragma unroll
            for (int jj = 0; jj < kKeyTile / 16; jj++) {
#pragma unroll
                for (int dt = 0; dt < D / 16; dt++) {
                    uint32_t b0, b1, b2, b3;
                    const int row = jj * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                    const int col = dt * 16 + (lane >> 4) * 8;
                    ldmatrix_x4_trans(b0, b1, b2, b3, &sV[row * SROW + col]);
                    mma_16x8x8(o[2 * dt], pa[2 * jj][0], pa[2 * jj][1], b0);
                    mma_16x8x8(o[2 * dt], pa[2 * jj + 1][0], pa[2 * jj + 1][1], b1);
                    mma_16x8x8(o[2 * dt + 1], pa[2 * jj][0], pa[2 * jj][1], b2);
                    mma_16x8x8(o[2 * dt + 1], pa[2 * jj + 1][0], pa[2 * jj + 1][1], b3);
                }
            }
        }
        __syncthreads();
    }

    if (!warp_active) return;
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int d = nt * 8 + 2 * tq;
        if (r0 < t_end)
            *reinterpret_cast<half2 *>(a.out + (static_cast<size_t>(r0) * a.n_heads + head) * D + d) =
                __floats2half2_rn(o[nt][0] * inv0, o[nt][1] * inv0);
        if (r1 < t_end)
            *reinterpret_cast<half2 *>(a.out + (static_cast<size_t>(r1) * a.n_heads + head) * D + d) =
                __floats2half2_rn(o[nt][2] * inv1, o[nt][3] * inv1);
    }
}

template <int D>
void launch_paged(const AttentionArgs &a, cudaStream_t s) {
    dim3 grid(a.T, a.n_kv, a.num_splits);
    switch (a.n_heads / a.n_kv) {
        case 1: paged_attention_kernel<D, 1><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 2: paged_attention_kernel<D, 2><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 3: paged_attention_kernel<D, 3><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 4: paged_attention_kernel<D, 4><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 5: paged_attention_kernel<D, 5><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 6: paged_attention_kernel<D, 6><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 7: paged_attention_kernel<D, 7><<<grid, kWarps * 32, 0, s>>>(a); break;
        case 8: paged_attention_kernel<D, 8><<<grid, kWarps * 32, 0, s>>>(a); break;
        default: fail("query heads per kv head must be 1..8 (got {})", a.n_heads / a.n_kv);
    }
}

}  // namespace

void prefill_attention(const PrefillArgs &a, cudaStream_t s) {
    if (a.num_tiles <= 0) return;
    dim3 grid(a.num_tiles, a.n_heads);
    switch (a.head_dim) {
        case 64: prefill_attention_kernel<64><<<grid, 128, 0, s>>>(a); break;
        case 128: prefill_attention_kernel<128><<<grid, 128, 0, s>>>(a); break;
        default: fail("prefill attention supports head_dim 64 and 128 (got {})", a.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
}

void paged_attention(const AttentionArgs &a, cudaStream_t s) {
    if (a.T <= 0) return;
    EMBER_CHECK(a.num_splits >= 1 && a.split_len % kBlockSize == 0, "attention: split_len must be a multiple of the block size");
    switch (a.head_dim) {
        case 64: launch_paged<64>(a, s); break;
        case 128: launch_paged<128>(a, s); break;
        case 256: launch_paged<256>(a, s); break;
        default: fail("head_dim {} is not supported", a.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    if (a.num_splits > 1) {
        combine_splits_kernel<<<dim3(a.T, a.n_heads), a.head_dim, 0, s>>>(a);
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace ember::cuda
