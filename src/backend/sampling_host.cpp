#include "backend/sampling_host.hpp"

#include "backend/sampling_rng.hpp"

#include <algorithm>
#include <cmath>

namespace ember {

double sample_uniform(uint64_t seed, uint64_t counter) {
    // splitmix64 over a mix of both inputs: well distributed, and stateless.
    uint64_t z = seed ^ (counter * 0x9E3779B97F4A7C15ull) ^ 0xD1B54A32D192ED03ull;
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);  // 53 bits
}

std::vector<std::pair<int32_t, float>> sampling_distribution(const float *logits, int vocab, const SamplingParams &p) {
    std::vector<std::pair<int32_t, float>> out;
    if (p.temperature <= 1e-5f) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best]) best = i;
        out.emplace_back(best, 1.0f);
        return out;
    }
    std::vector<std::pair<float, int32_t>> cand;
    cand.reserve(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; i++) cand.emplace_back(logits[i] / p.temperature, i);
    auto by_logit = [](const auto &a, const auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
    size_t keep = cand.size();
    if (p.top_k > 0 && static_cast<size_t>(p.top_k) < keep) {
        keep = static_cast<size_t>(p.top_k);
        std::nth_element(cand.begin(), cand.begin() + static_cast<ptrdiff_t>(keep - 1), cand.end(), by_logit);
        cand.resize(keep);
    }
    std::sort(cand.begin(), cand.end(), by_logit);

    double max_l = cand[0].first, total = 0;
    std::vector<double> probs(cand.size());
    for (size_t i = 0; i < cand.size(); i++) total += probs[i] = std::exp(static_cast<double>(cand[i].first) - max_l);
    for (double &x : probs) x /= total;

    size_t n = cand.size();
    if (p.top_p < 1.0f) {
        double cum = 0;
        for (size_t i = 0; i < n; i++) {
            cum += probs[i];
            if (cum >= p.top_p) {
                n = i + 1;
                break;
            }
        }
    }
    if (p.min_p > 0.0f) {
        double floor = p.min_p * probs[0];
        size_t m = 1;
        while (m < n && probs[m] >= floor) m++;
        n = m;
    }
    double kept = 0;
    for (size_t i = 0; i < n; i++) kept += probs[i];
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(cand[i].second, static_cast<float>(probs[i] / kept));
    return out;
}

int32_t sample_host(const float *logits, int vocab, const SampleRequest &req) {
    auto dist = sampling_distribution(logits, vocab, req.params);
    if (dist.size() == 1) return dist[0].first;
    // Gumbel-max over the kept tokens: the same noise the GPU sampler draws.
    int32_t best = dist[0].first;
    float best_score = -INFINITY;
    for (const auto &[tok, prob] : dist) {
        float score = logits[tok] / req.params.temperature +
                      gumbel_noise(req.params.seed, req.counter, static_cast<uint32_t>(tok));
        if (score > best_score) {
            best_score = score;
            best = tok;
        }
    }
    return best;
}

}  // namespace ember
