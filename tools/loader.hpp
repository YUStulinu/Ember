// Loading the model(s) for a command: the tokenizer, the backend and, for
// speculative decoding, a draft model sharing the GPU.
#pragma once

#include <algorithm>
#include <chrono>
#include <memory>

#include "backend/backend.hpp"
#include "cli_args.hpp"
#include "common/log.hpp"
#include "model/tokenizer.hpp"

namespace ember::cli {

struct Loaded {
    std::unique_ptr<Backend> backend, draft;
    Tokenizer tokenizer;
};

inline Loaded load(const cli::Options &opt) {
    auto t0 = std::chrono::steady_clock::now();
    Loaded l;
    l.tokenizer = Tokenizer::load(opt.model + "/tokenizer.json");
    BackendOptions bo = opt.backend;
    if (!opt.draft.empty()) {
        // Speculative decoding: the target checks k + 1 positions per sequence,
        // and the two models share the GPU, so each gets part of the KV memory.
        // (capped: 256 rows of 150k-entry logits are already 150 MB)
        bo.max_logit_rows = std::min(bo.max_seqs * (opt.spec_tokens + 1), 256);
        BackendOptions dop = opt.backend;
        if (!opt.draft_weights.empty()) dop.weights = parse_weight_format(opt.draft_weights);
        if (bo.kv_cache_bytes == 0 && bo.device == "cuda") {
            const int64_t share = plan_kv_split(opt.model, opt.draft, bo, dop);
            bo.kv_cache_bytes = dop.kv_cache_bytes = share;
        }
        l.backend = create_backend(opt.model, bo);
        l.draft = create_backend(opt.draft, dop);
    } else {
        l.backend = create_backend(opt.model, bo);
    }
    log::info("loaded {}{} in {:.1f} s", opt.model, opt.draft.empty() ? "" : " + draft " + opt.draft, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return l;
}

}  // namespace ember::cli
