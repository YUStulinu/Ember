// Sampling on the GPU, one block per row of logits.
//
// The kept set (top-k, top-p, min-p) is described by a single logit threshold:
// a token is kept iff x_i >= theta. Thresholds are found by radix selection
// over the float bits (4 passes of 8 bits) instead of sorting 150k values:
//   top-k: the k-th largest x (weights 1, target k),
//   top-p: the smallest x whose cumulative probability, going down from the
//          top, reaches top_p (weights exp(x - max), target top_p * Z).
// The token is then argmax over kept i of x_i + gumbel_i.
#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/kernels.cuh"
#include "backend/sampling_rng.hpp"

namespace ember::cuda {

namespace {

constexpr int kThreads = 1024;

__device__ __forceinline__ uint32_t float_key(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float key_float(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

struct Shared {
    float hist[256];
    float scratch[32];
    uint32_t prefix, mask;
    float above;
    int best_idx[32];
    float best_val[32];
};

// The largest key v such that the weight of elements with key >= v reaches
// `target`, among elements with x >= floor. Weight is 1, or exp(x - max) when
// `prob_weights`.
__device__ float radix_threshold(const float *x, int V, float inv_t, float floor, float mx, bool prob_weights,
                                 float target, Shared &sh) {
    if (threadIdx.x == 0) {
        sh.prefix = 0;
        sh.mask = 0;
        sh.above = 0.0f;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) sh.hist[i] = 0.0f;
        __syncthreads();
        const uint32_t prefix = sh.prefix, mask = sh.mask;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float v = x[i] * inv_t;
            if (v < floor) continue;
            uint32_t k = float_key(v);
            if ((k & mask) != prefix) continue;
            atomicAdd(&sh.hist[(k >> shift) & 255u], prob_weights ? __expf(v - mx) : 1.0f);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float cum = sh.above;
            int d = 255;
            for (; d > 0; d--) {
                if (cum + sh.hist[d] >= target) break;
                cum += sh.hist[d];
            }
            sh.above = cum;
            sh.prefix = prefix | (static_cast<uint32_t>(d) << shift);
            sh.mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    return key_float(sh.prefix);
}

// Computes the max (scaled) logit, the kept-set threshold and the normalizer
// of the kept set. Returns false for greedy rows.
__device__ bool distribution(const float *x, int V, const DeviceSampleParams &p, Shared &sh, float &mx, float &theta,
                             float &z) {
    if (p.temperature <= 1e-5f) return false;
    const float inv_t = 1.0f / p.temperature;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, x[i] * inv_t);
    mx = block_max(m, sh.scratch);

    theta = -INFINITY;
    if (p.top_k > 0 && p.top_k < V) theta = radix_threshold(x, V, inv_t, -INFINITY, mx, false, static_cast<float>(p.top_k), sh);
    if (p.min_p > 0.0f) theta = fmaxf(theta, mx + logf(p.min_p));

    float s = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = x[i] * inv_t;
        if (v >= theta) s += __expf(v - mx);
    }
    z = block_sum(s, sh.scratch);

    if (p.top_p < 1.0f) {
        float tp = radix_threshold(x, V, inv_t, theta, mx, true, p.top_p * z, sh);
        if (tp > theta) {
            theta = tp;
            s = 0.0f;
            for (int i = threadIdx.x; i < V; i += blockDim.x) {
                float v = x[i] * inv_t;
                if (v >= theta) s += __expf(v - mx);
            }
            z = block_sum(s, sh.scratch);
        }
    }
    return true;
}

// Block-wide argmax of score(i); ties go to the lowest index.
template <class F>
__device__ int block_argmax(int V, F score, Shared &sh) {
    float best = -INFINITY;
    int idx = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = score(i);
        if (v > best || (v == best && i < idx)) {
            best = v;
            idx = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > best || (ov == best && oi < idx)) {
            best = ov;
            idx = oi;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) {
        sh.best_val[warp] = best;
        sh.best_idx[warp] = idx;
    }
    __syncthreads();
    best = -INFINITY;
    idx = 0x7FFFFFFF;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); w++) {
        if (sh.best_val[w] > best || (sh.best_val[w] == best && sh.best_idx[w] < idx)) {
            best = sh.best_val[w];
            idx = sh.best_idx[w];
        }
    }
    return idx == 0x7FFFFFFF ? 0 : idx;
}

__global__ void __launch_bounds__(kThreads) sample_kernel(const float *logits, int V, const DeviceSampleParams *params, int32_t *out) {
    __shared__ Shared sh;
    const int r = blockIdx.x;
    const float *x = logits + static_cast<size_t>(r) * V;
    const DeviceSampleParams p = params[r];
    float mx, theta, z;
    int tok;
    if (!distribution(x, V, p, sh, mx, theta, z)) {
        tok = block_argmax(V, [&](int i) { return x[i]; }, sh);
    } else {
        const float inv_t = 1.0f / p.temperature;
        tok = block_argmax(V, [&](int i) {
            float v = x[i] * inv_t;
            return v >= theta ? v + gumbel_noise(p.seed, p.counter, static_cast<uint32_t>(i)) : -INFINITY;
        }, sh);
    }
    if (threadIdx.x == 0) out[r] = tok;
}

__global__ void __launch_bounds__(kThreads) prob_kernel(const float *logits, int V, const DeviceSampleParams *params,
                                                        const int32_t *rows, const int32_t *tokens, float *out) {
    __shared__ Shared sh;
    const int q = blockIdx.x, r = rows[q], tok = tokens[q];
    const float *x = logits + static_cast<size_t>(r) * V;
    const DeviceSampleParams p = params[r];
    float mx, theta, z;
    float prob;
    if (!distribution(x, V, p, sh, mx, theta, z)) {
        prob = block_argmax(V, [&](int i) { return x[i]; }, sh) == tok ? 1.0f : 0.0f;
    } else {
        float v = x[tok] / p.temperature;
        prob = v >= theta ? __expf(v - mx) / z : 0.0f;
    }
    if (threadIdx.x == 0) out[q] = prob;
}

}  // namespace

void sample_tokens(const float *logits, int R, int V, const DeviceSampleParams *params, int32_t *out, cudaStream_t s) {
    if (R <= 0) return;
    sample_kernel<<<R, kThreads, 0, s>>>(logits, V, params, out);
    CUDA_CHECK(cudaGetLastError());
}

void token_probabilities(const float *logits, int V, const DeviceSampleParams *params, const int32_t *rows,
                         const int32_t *tokens, int n, float *out, cudaStream_t s) {
    if (n <= 0) return;
    prob_kernel<<<n, kThreads, 0, s>>>(logits, V, params, rows, tokens, out);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ember::cuda
