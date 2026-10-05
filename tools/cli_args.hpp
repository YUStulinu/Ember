// Command-line options shared by the ember subcommands.
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend/backend.hpp"
#include "common/error.hpp"
#include "common/mmap_file.hpp"
#include "engine/engine.hpp"

namespace ember::cli {

struct Options {
    std::string model;
    std::string draft;  // speculative decoding draft model
    std::string draft_weights;  // its weight format (default: same as --weights)
    BackendOptions backend;
    EngineOptions engine;
    SamplingParams sampling;
    bool sampling_set = false;
    int max_tokens = 512;
    bool think = false;
    bool raw = false;
    bool verbose = false;
    std::string prompt;
    std::string system;
    std::string host = "127.0.0.1";
    int port = 8080;
    int spec_tokens = 4;
    std::vector<std::string> rest;  // positional arguments
};

inline std::string default_model() {
    for (const char *m : {"models/Qwen3-1.7B", "models/Qwen3-0.6B", "../models/Qwen3-1.7B"})
        if (file_exists(std::string(m) + "/config.json")) return m;
    return "models/Qwen3-1.7B";
}

// Parses argv[first..]; throws ember::Error on unknown options.
inline Options parse(int argc, char **argv, int first) {
    Options o;
    o.backend.device = cuda_available() ? "cuda" : "cpu";
    // Qwen3's recommended settings for non-thinking chat.
    o.sampling.temperature = 0.7f;
    o.sampling.top_p = 0.8f;
    o.sampling.top_k = 20;
    for (int i = first; i < argc; i++) {
        std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) fail("option {} needs a value", a);
            return argv[++i];
        };
        auto num = [&]() { return std::atof(value().c_str()); };
        if (a == "-m" || a == "--model") o.model = value();
        else if (a == "--draft") o.draft = value();
        else if (a == "--draft-weights") o.draft_weights = value();
        else if (a == "--spec-tokens") o.spec_tokens = o.engine.spec_tokens = static_cast<int>(num());
        else if (a == "--device") o.backend.device = value();
        else if (a == "--weights" || a == "-q") o.backend.weights = parse_weight_format(value());
        else if (a == "--quant-mix") o.backend.quant_mix = value();
        else if (a == "--threads") o.backend.threads = static_cast<int>(num());
        else if (a == "--max-batch-tokens") o.backend.max_batch_tokens = static_cast<int>(num());
        else if (a == "--max-seqs") o.backend.max_seqs = static_cast<int>(num());
        else if (a == "--gpu-memory") o.backend.memory_fraction = num();
        else if (a == "--kv-cache-mib") o.backend.kv_cache_bytes = static_cast<int64_t>(num() * 1048576.0);
        else if (a == "--kv-cache-tokens") o.backend.kv_cache_tokens = static_cast<int>(num());
        else if (a == "--no-cuda-graphs") o.backend.cuda_graphs = false;
        else if (a == "--no-prefix-cache") o.engine.prefix_caching = false;
        else if (a == "--prefill-chunk") o.engine.max_prefill_chunk = static_cast<int>(num());
        else if (a == "-n" || a == "--max-tokens") o.max_tokens = static_cast<int>(num());
        else if (a == "--temp" || a == "--temperature") { o.sampling.temperature = static_cast<float>(num()); o.sampling_set = true; }
        else if (a == "--top-p") { o.sampling.top_p = static_cast<float>(num()); o.sampling_set = true; }
        else if (a == "--top-k") { o.sampling.top_k = static_cast<int>(num()); o.sampling_set = true; }
        else if (a == "--min-p") { o.sampling.min_p = static_cast<float>(num()); o.sampling_set = true; }
        else if (a == "--seed") o.sampling.seed = std::strtoull(value().c_str(), nullptr, 10);
        else if (a == "--greedy") { o.sampling.temperature = 0; o.sampling_set = true; }
        else if (a == "--think") o.think = true;
        else if (a == "--raw") o.raw = true;
        else if (a == "-p" || a == "--prompt") o.prompt = value();
        else if (a == "-s" || a == "--system") o.system = value();
        else if (a == "--host") o.host = value();
        else if (a == "--port") o.port = static_cast<int>(num());
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else if (!a.empty() && a[0] == '-') fail("unknown option {} (see ember help)", a);
        else o.rest.push_back(a);
    }
    if (o.model.empty()) o.model = default_model();
    if (o.think && !o.sampling_set) {  // Qwen3's recommended settings for thinking mode
        o.sampling.temperature = 0.6f;
        o.sampling.top_p = 0.95f;
    }
    return o;
}

}  // namespace ember::cli
