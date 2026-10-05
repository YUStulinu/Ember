// The random numbers behind sampling, shared by host and device code so both
// backends draw the same noise.
//
// Sampling uses the Gumbel-max trick: argmax_i (x_i + g_i) with g_i Gumbel
// noise is distributed exactly as softmax(x). Each g_i depends only on
// (seed, counter, token id), so it needs no sorting, parallelizes over the
// vocabulary, and a sequence draws the same tokens whatever shares its batch.
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define EMBER_HD __host__ __device__ __forceinline__
#else
#define EMBER_HD inline
#endif

namespace ember {

EMBER_HD uint64_t rng_mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Gumbel(0, 1) noise for token `id` at draw `counter` of a sequence with `seed`.
EMBER_HD float gumbel_noise(uint64_t seed, uint64_t counter, uint32_t id) {
    uint64_t h = rng_mix(seed ^ rng_mix(counter * 0x632BE59BD9B4E019ull + id));
    float u = (static_cast<float>(h >> 40) + 0.5f) * (1.0f / 16777216.0f);  // (0, 1)
    return -logf(-logf(u));
}

}  // namespace ember
