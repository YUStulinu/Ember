// A fake backend for engine and server tests: its "model" reads the context
// back from the paged cache (following the block tables), so any scheduling
// or block-accounting mistake changes the generated tokens.
#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "backend/backend.hpp"
#include "test.hpp"

namespace ember::test {

constexpr int kVocab = 1000;
constexpr int kEos = 1;

// Next token after a context: a hash of every token, never EOS unless asked.
inline int32_t next_token(const std::vector<int32_t> &ctx, bool allow_eos, int disagree = 0) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : ctx) {
        h ^= static_cast<uint64_t>(t);
        h *= 1099511628211ull;
    }
    int32_t tok = static_cast<int32_t>(2 + (h >> 17) % (kVocab - 2));
    if (allow_eos && (h >> 7) % 23 == 0) tok = kEos;
    if (disagree && (h >> 40) % disagree == 0) tok = static_cast<int32_t>(2 + (tok + 7) % (kVocab - 2));  // a wrong guess
    return tok;
}

class FakeBackend final : public Backend {
public:
    // `vocab`: the advertised vocabulary (prompts may use any id below it); generated ids stay below kVocab.
    FakeBackend(int blocks, int max_batch_tokens, int max_seqs, bool eos, int disagree = 0, int logit_rows = 0,
                int vocab = kVocab)
        : eos_(eos), disagree_(disagree) {
        cfg_.arch = "qwen3";
        cfg_.vocab_size = vocab;
        cfg_.hidden = 8;
        cfg_.n_layers = 1;
        cfg_.n_heads = cfg_.n_kv_heads = 1;
        cfg_.head_dim = 8;
        cfg_.max_position = 100000;
        cfg_.eos_ids = {kEos};
        opt_.max_batch_tokens = max_batch_tokens;
        opt_.max_seqs = max_seqs;
        opt_.max_logit_rows = logit_rows ? logit_rows : max_seqs;
        cache_.assign(static_cast<size_t>(blocks) * kBlockSize, -1);
        blocks_ = blocks;
    }
    std::string name() const override { return "fake"; }
    const ModelConfig &config() const override { return cfg_; }
    const BackendOptions &options() const override { return opt_; }
    int num_kv_blocks() const override { return blocks_; }
    MemoryInfo memory() const override { return {}; }

    void forward(const StepBatch &b) override {
        CHECK(b.num_tokens() <= opt_.max_batch_tokens && b.num_seqs() <= opt_.max_seqs);
        CHECK(static_cast<int>(b.logit_rows.size()) <= opt_.max_logit_rows);
        steps++;
        max_seen_tokens = std::max(max_seen_tokens, b.num_tokens());
        // Write every new token into its slot first (as the real backends do).
        for (int s = 0; s < b.num_seqs(); s++)
            for (int t = b.query_start[s]; t < b.query_start[s + 1]; t++)
                cache_[slot(b, s, b.positions[static_cast<size_t>(t)])] = b.tokens[static_cast<size_t>(t)];
        next_.clear();
        for (int32_t row : b.logit_rows) {
            int s = 0;
            while (b.query_start[s + 1] <= row) s++;
            std::vector<int32_t> ctx;
            for (int p = 0; p <= b.positions[static_cast<size_t>(row)]; p++) ctx.push_back(cache_[slot(b, s, p)]);
            next_.push_back(next_token(ctx, eos_, disagree_));
        }
    }
    void sample(std::span<const SampleRequest> reqs, std::span<int32_t> out) override {
        CHECK(reqs.size() == next_.size());
        for (size_t i = 0; i < out.size(); i++) out[i] = next_[i];
    }
    void token_probs(std::span<const SampleRequest>, std::span<const int32_t>, std::span<const int32_t>, std::span<float>) override {}
    void logits(int, std::span<float>) override {}
    void set_capture_hidden(bool) override {}
    std::vector<float> captured_hidden() override { return {}; }

    int steps = 0, max_seen_tokens = 0;

private:
    size_t slot(const StepBatch &b, int s, int pos) const {
        int block = b.block_tables[static_cast<size_t>(s) * b.max_blocks + pos / kBlockSize];
        CHECK(block >= 0 && block < blocks_);
        return static_cast<size_t>(block) * kBlockSize + pos % kBlockSize;
    }
    ModelConfig cfg_;
    BackendOptions opt_;
    std::vector<int32_t> cache_;
    std::vector<int32_t> next_;
    int blocks_;
    bool eos_;
    int disagree_;
};

// What a request must produce, computed directly.
inline std::vector<int32_t> expected(std::vector<int32_t> ctx, int max_tokens, bool eos) {
    std::vector<int32_t> out;
    for (int i = 0; i < max_tokens; i++) {
        int32_t t = next_token(ctx, eos);
        if (t == kEos) break;
        out.push_back(t);
        ctx.push_back(t);
    }
    return out;
}

inline std::vector<int32_t> random_tokens(std::mt19937 &rng, int n) {
    std::uniform_int_distribution<int32_t> d(2, kVocab - 1);
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (auto &t : v) t = d(rng);
    return v;
}


}  // namespace ember::test
