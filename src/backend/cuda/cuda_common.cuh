// Shared helpers for the CUDA backend: error checking, device buffers, and
// the small PTX wrappers the kernels are built from.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#include "common/error.hpp"

namespace ember::cuda {

#define CUDA_CHECK(expr)                                                                                    \
    do {                                                                                                    \
        cudaError_t err_ = (expr);                                                                          \
        if (err_ != cudaSuccess)                                                                            \
            ::ember::fail("CUDA error at {}:{}: {} ({})", __FILE__, __LINE__, cudaGetErrorString(err_), #expr); \
    } while (0)

// Owning device allocation.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { resize(count); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
            count_ = std::exchange(o.count_, 0);
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void resize(size_t count) {
        release();
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
        count_ = count;
    }
    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }
    T *get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

// ---- device-side helpers ------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the whole block; `scratch` needs one float per warp. All threads get the result.
__device__ __forceinline__ float block_sum(float v, float *scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0;
    for (int w = 0; w < warps; w++) total += scratch[w];
    return total;
}

__device__ __forceinline__ float block_max(float v, float *scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int w = 0; w < warps; w++) m = fmaxf(m, scratch[w]);
    return m;
}

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 matrices of 16-bit elements from shared memory; thread t gives the
// address of row (t % 8) of matrix (t / 8).
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(smem_addr(p)));
}

// D = A * B + D for a 16x8x8 tile: A row-major f16 (2 regs), B column-major
// f16 (1 reg), C/D f32 (4 regs). Available from sm_75 (Turing) on.
__device__ __forceinline__ void mma_16x8x8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t *>(&h);
}

// SplitMix64 finalizer: a stateless hash, used for counter-based random numbers.
__host__ __device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace ember::cuda
