// Token sampling on the host: the reference implementation (used by the CPU
// backend and by tests that check the GPU sampler).
#pragma once

#include <cstdint>
#include <vector>

#include "backend/backend.hpp"

namespace ember {

// Uniform number in [0, 1) that depends only on (seed, counter).
double sample_uniform(uint64_t seed, uint64_t counter);

// Turns logits into the sampling distribution: temperature, then top-k, top-p
// and min-p filtering. Returns (token, probability) pairs of the kept tokens,
// most likely first. Greedy params give the single best token with p = 1.
std::vector<std::pair<int32_t, float>> sampling_distribution(const float *logits, int vocab, const SamplingParams &p);

// Draws one token.
int32_t sample_host(const float *logits, int vocab, const SampleRequest &request);

}  // namespace ember
