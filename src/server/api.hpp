// The HTTP API on top of the engine:
//   POST /v1/chat/completions   OpenAI Chat Completions (streaming with SSE)
//   POST /v1/completions        OpenAI legacy completions (raw prompt)
//   POST /v1/messages           Anthropic Messages (streaming with SSE events)
//   GET  /v1/models             the served model
//   GET  /health, /metrics      liveness, Prometheus metrics
//   GET  /api/stats, /api/events   dashboard data (JSON / SSE)
//   GET  /                      the dashboard
#pragma once

#include <chrono>
#include <string>

#include "backend/backend.hpp"
#include "engine/engine.hpp"
#include "model/tokenizer.hpp"
#include "server/http.hpp"

namespace ember {

struct ApiContext {
    Engine &engine;
    const Tokenizer &tokenizer;
    std::string model_name;
    std::string backend_name;
    std::string weights;
    SamplingParams default_sampling;
    int default_max_tokens = 1024;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
};

void register_api(http::Server &server, ApiContext &ctx);

}  // namespace ember
