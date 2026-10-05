// Drives a backend directly for one sequence (no engine): fills the block
// table with consecutive blocks, prefills, then decodes greedily.
#pragma once

#include <vector>

#include "backend/backend.hpp"

namespace ember::test {

class SingleSequence {
public:
    explicit SingleSequence(Backend &be, int first_block = 0) : be_(be), first_block_(first_block) {}

    // Runs the prompt (optionally in chunks) and leaves its last logits as row 0.
    void prefill(const std::vector<int> &prompt, int chunk = 0) {
        if (chunk <= 0) chunk = static_cast<int>(prompt.size());
        for (size_t start = 0; start < prompt.size(); start += static_cast<size_t>(chunk)) {
            size_t end = std::min(prompt.size(), start + static_cast<size_t>(chunk));
            step(std::vector<int>(prompt.begin() + static_cast<ptrdiff_t>(start), prompt.begin() + static_cast<ptrdiff_t>(end)));
        }
    }

    // Generates n tokens greedily (the first from the last prefill's logits).
    std::vector<int> greedy(int n) {
        std::vector<int> out;
        SampleRequest req;
        req.params.temperature = 0;
        for (int i = 0; i < n; i++) {
            int32_t tok;
            be_.sample({&req, 1}, {&tok, 1});
            out.push_back(tok);
            if (i + 1 < n) step({tok});
        }
        return out;
    }

    void step(const std::vector<int> &tokens) {
        StepBatch b;
        b.clear();
        for (size_t i = 0; i < tokens.size(); i++) {
            b.tokens.push_back(tokens[i]);
            b.positions.push_back(len_ + static_cast<int>(i));
        }
        len_ += static_cast<int>(tokens.size());
        b.query_start.push_back(static_cast<int32_t>(tokens.size()));
        b.context_len.push_back(len_);
        b.max_blocks = (len_ + kBlockSize - 1) / kBlockSize;
        for (int i = 0; i < b.max_blocks; i++) b.block_tables.push_back(first_block_ + i);
        b.logit_rows.push_back(static_cast<int32_t>(tokens.size()) - 1);
        be_.forward(b);
    }

    int length() const { return len_; }

private:
    Backend &be_;
    int first_block_;
    int len_ = 0;
};

}  // namespace ember::test
