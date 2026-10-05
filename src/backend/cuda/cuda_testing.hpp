// Host-friendly entry points into the CUDA kernels, for the unit tests
// (which are compiled by the C++ compiler, not nvcc).
#pragma once

#include <cstdint>
#include <vector>

#include "backend/backend.hpp"

namespace ember::cuda {

// C = A * W^T with A [M, K] and W [N, K] given as f16 bits. `epilogue` and
// `kernel` are the numeric values of cuda::Epilogue and cuda::GemmKernel.
// For epilogue add_f32 the output starts at 1.0. Returns C as floats.
std::vector<float> test_gemm(const std::vector<uint16_t> &A, const std::vector<uint16_t> &W, int M, int N, int K,
                             int epilogue, int kernel, bool split_k = false);  // split_k: allow splitting K

// Average time in microseconds of one C = A * W^T with random data (kernel
// numbered as cuda::GemmKernel; split_k as in test_gemm).
double bench_gemm(int M, int N, int K, int epilogue, int kernel, bool split_k, int iters);

// Quantizes W ([N, K] f16 bits) with `qt` (1 int8, 2 int4), then computes
// C = A * W^T twice: with the quantized decode kernel, and with the f16 GEMM on
// the dequantized weights. Returns {quantized, reference, dequantized W}.
struct QuantTestResult {
    std::vector<float> quant, reference, dequantized;
};
QuantTestResult test_quant_gemm(const std::vector<uint16_t> &A, const std::vector<uint16_t> &W, int M, int N, int K, int qt,
                                int epilogue);

// The GPU sampler on host logits [R, V].
std::vector<int32_t> test_sample(const std::vector<float> &logits, int R, int V, const std::vector<SampleRequest> &reqs);

}  // namespace ember::cuda
