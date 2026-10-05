// The HTTP API on top of the engine:
//   POST /v1/chat/completions   OpenAI Chat Completions (streaming with SSE)
//   POST /v1/completions        OpenAI legacy completions (raw prompt)
//   POST /v1/messages           Anthropic Messages (streaming with SSE events)
//   GET  /v1/models             the served model
//   GET  /health, /metrics      liveness, Prometheus metrics
//   GET  /api/stats, /api/events, /api/history   dashboard data
//   GET  /                      the dashboard
#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "backend/backend.hpp"
#include "engine/engine.hpp"
#include "model/tokenizer.hpp"
#include "server/http.hpp"

namespace ember {

// The last two minutes of throughput, sampled twice a second, so the
// dashboard's chart is full as soon as the page opens.
struct ThroughputHistory {
    static constexpr size_t kSamples = 240;
    std::mutex mutex;
    std::deque<std::pair<float, float>> samples;  // (generated, prompt) tokens/s
    std::atomic<bool> stop{false};
    std::thread thread;
    ~ThroughputHistory() {
        stop = true;
        if (thread.joinable()) thread.join();
    }
};

struct ApiContext {
    ApiContext(Engine &e, const Tokenizer &t) : engine(e), tokenizer(t) {}
    Engine &engine;
    const Tokenizer &tokenizer;
    std::string model_name;
    std::string backend_name;
    std::string weights;
    SamplingParams default_sampling;
    int default_max_tokens = 1024;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::unique_ptr<ThroughputHistory> history;
};

void register_api(http::Server &server, ApiContext &ctx);

}  // namespace ember
