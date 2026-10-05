// Weight conversion, embedding lookup, RMSNorm, and the fused
// QK-norm + RoPE + KV-cache write.
#include "backend/backend.hpp"
#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/kernels.cuh"

namespace ember::cuda {

namespace {

constexpr float kHalfMax = 65504.0f;

__device__ __forceinline__ float load_src(const void *src, SrcType t, int64_t i) {
    switch (t) {
        case SrcType::f32: return static_cast<const float *>(src)[i];
        case SrcType::f16: return __half2float(static_cast<const half *>(src)[i]);
        default: {
            uint32_t bits = static_cast<uint32_t>(static_cast<const uint16_t *>(src)[i]) << 16;
            return __uint_as_float(bits);
        }
    }
}

__global__ void convert_rows_kernel(const void *src, SrcType type, int64_t rows, int64_t cols, half *dst,
                                    int64_t dst_offset, int64_t dst_stride, unsigned long long *overflow) {
    int64_t n = rows * cols;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t r = i / cols, c = i % cols;
        float v = load_src(src, type, i);
        if (fabsf(v) > kHalfMax) {
            atomicAdd(overflow, 1ull);
            v = fminf(fmaxf(v, -kHalfMax), kHalfMax);
        }
        dst[(dst_offset + r * dst_stride) * cols + c] = __float2half_rn(v);
    }
}

__global__ void convert_f32_kernel(const void *src, SrcType type, int64_t n, float *dst) {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = load_src(src, type, i);
}

__global__ void embed_kernel(const int32_t *tokens, const half *table, int hidden, float *out) {
    const int t = blockIdx.x;
    const half2 *row = reinterpret_cast<const half2 *>(table + static_cast<size_t>(tokens[t]) * hidden);
    float2 *o = reinterpret_cast<float2 *>(out + static_cast<size_t>(t) * hidden);
    for (int i = threadIdx.x; i < hidden / 2; i += blockDim.x) o[i] = __half22float2(row[i]);
}

__global__ void rmsnorm_kernel(const float *x, const int32_t *rows, int hidden, const float *w, float eps, half *out) {
    __shared__ float scratch[32];
    const int r = blockIdx.x;
    const float *xr = x + static_cast<size_t>(rows ? rows[r] : r) * hidden;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, scratch);
    const float scale = rsqrtf(ss / hidden + eps);
    half *o = out + static_cast<size_t>(r) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = w[i] * (xr[i] * scale);
        o[i] = __float2half_rn(fminf(fmaxf(v, -kHalfMax), kHalfMax));
    }
}

// One warp per head vector. Lane l holds elements [l * E, l * E + E) with
// E = D / 32; the rotary partner of element d is d +- D/2, which lives in lane
// l ^ 16 - one shuffle away.
template <int D>
__global__ void qk_norm_rope_store_kernel(QkvArgs a) {
    constexpr int E = D / 32;
    const int t = blockIdx.x;
    const int head = blockIdx.y * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int total = a.n_heads + 2 * a.n_kv;
    if (head >= total) return;

    const half *src = a.qkv + static_cast<size_t>(t) * total * D + static_cast<size_t>(head) * D + lane * E;
    float x[E];
#pragma unroll
    for (int i = 0; i < E; i++) x[i] = __half2float(src[i]);
    if (a.bias) {
#pragma unroll
        for (int i = 0; i < E; i++) x[i] += a.bias[head * D + lane * E + i];
    }

    const int pos = a.positions[t];
    const bool is_q = head < a.n_heads, is_k = !is_q && head < a.n_heads + a.n_kv;
    if (is_q || is_k) {
        const float *nw = is_q ? a.q_norm : a.k_norm;
        if (nw) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < E; i++) ss += x[i] * x[i];
            ss = warp_sum(ss);
            const float scale = rsqrtf(ss / D + a.eps);
#pragma unroll
            for (int i = 0; i < E; i++) x[i] = nw[lane * E + i] * (x[i] * scale);
        }
        // Rotate-half RoPE: (x_d, x_{d+D/2}) rotated by pos * inv_freq[d mod D/2].
        const bool low = lane < 16;
#pragma unroll
        for (int i = 0; i < E; i++) {
            float partner = __shfl_xor_sync(0xffffffffu, x[i], 16);
            int d = (lane * E + i) % (D / 2);
            float s, c;
            sincosf(static_cast<float>(pos) * a.inv_freq[d], &s, &c);
            x[i] = low ? x[i] * c - partner * s : x[i] * c + partner * s;
        }
    }

    if (is_q) {
        half *dst = a.q_out + static_cast<size_t>(t) * a.n_heads * D + static_cast<size_t>(head) * D + lane * E;
#pragma unroll
        for (int i = 0; i < E; i++) dst[i] = __float2half_rn(x[i]);
        return;
    }
    const int kvh = is_k ? head - a.n_heads : head - a.n_heads - a.n_kv;
    const int seq = a.token_seq[t];
    const int block = a.block_tables[static_cast<size_t>(seq) * a.max_blocks + pos / kBlockSize];
    half *cache = is_k ? a.k_cache : a.v_cache;
    half *dst = cache + ((static_cast<size_t>(block) * a.n_kv + kvh) * kBlockSize + pos % kBlockSize) * D + lane * E;
#pragma unroll
    for (int i = 0; i < E; i++) dst[i] = __float2half_rn(fminf(fmaxf(x[i], -kHalfMax), kHalfMax));
}

int grid_for(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return static_cast<int>(blocks < 65535 * 8 ? blocks : 65535 * 8);
}

}  // namespace

void convert_rows_to_f16(const void *src, SrcType type, int64_t rows, int64_t cols, half *dst, int64_t dst_offset,
                         int64_t dst_stride, unsigned long long *overflow, cudaStream_t s) {
    convert_rows_kernel<<<grid_for(rows * cols, 256), 256, 0, s>>>(src, type, rows, cols, dst, dst_offset, dst_stride, overflow);
    CUDA_CHECK(cudaGetLastError());
}

void convert_to_f32(const void *src, SrcType type, int64_t n, float *dst, cudaStream_t s) {
    convert_f32_kernel<<<grid_for(n, 256), 256, 0, s>>>(src, type, n, dst);
    CUDA_CHECK(cudaGetLastError());
}

void embed_tokens(const int32_t *tokens, int T, const half *table, int hidden, float *out, cudaStream_t s) {
    if (T <= 0) return;
    embed_kernel<<<T, 256, 0, s>>>(tokens, table, hidden, out);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_rows(const float *x, const int32_t *rows, int R, int hidden, const float *w, float eps, half *out,
                  cudaStream_t s) {
    if (R <= 0) return;
    rmsnorm_kernel<<<R, hidden >= 2048 ? 512 : 256, 0, s>>>(x, rows, hidden, w, eps, out);
    CUDA_CHECK(cudaGetLastError());
}

void qk_norm_rope_store(const QkvArgs &a, cudaStream_t s) {
    if (a.T <= 0) return;
    const int heads = a.n_heads + 2 * a.n_kv;
    const int warps_per_block = heads < 16 ? heads : 16;
    dim3 grid(a.T, ceil_div(heads, warps_per_block));
    switch (a.head_dim) {
        case 64: qk_norm_rope_store_kernel<64><<<grid, warps_per_block * 32, 0, s>>>(a); break;
        case 128: qk_norm_rope_store_kernel<128><<<grid, warps_per_block * 32, 0, s>>>(a); break;
        case 256: qk_norm_rope_store_kernel<256><<<grid, warps_per_block * 32, 0, s>>>(a); break;
        default: fail("head_dim {} is not supported by the CUDA backend (64, 128, 256)", a.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ember::cuda
