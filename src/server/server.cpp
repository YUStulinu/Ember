// `ember serve`: load the model, start the engine loop and the HTTP server.
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <thread>

#include "../../tools/cli_args.hpp"
#include "../../tools/loader.hpp"
#include "common/log.hpp"
#include "model/tokenizer.hpp"
#include "server/api.hpp"
#include "server/http.hpp"

namespace ember {

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }
}  // namespace

int run_server(const cli::Options &opt) {
    auto t0 = std::chrono::steady_clock::now();
    cli::Loaded l = cli::load(opt);
    Tokenizer &tok = l.tokenizer;
    Backend *backend = l.backend.get();
    EngineOptions eo = opt.engine;
    eo.spec_tokens = opt.spec_tokens;
    Engine engine(*backend, eo, l.draft.get());
    engine.start();

    ApiContext ctx{engine, tok};
    ctx.model_name = std::filesystem::path(opt.model).filename().string();
    if (ctx.model_name.empty()) ctx.model_name = opt.model;
    ctx.backend_name = backend->name();
    ctx.weights = weight_format_name(opt.backend.weights);
    ctx.default_sampling = opt.sampling;
    ctx.default_max_tokens = opt.max_tokens;

    http::Server server;
    register_api(server, ctx);
    server.start(opt.host, opt.port);
    log::info("ready in {:.1f} s: http://{}:{}/  (dashboard), API at /v1/chat/completions and /v1/messages",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              opt.host == "0.0.0.0" ? "localhost" : opt.host, server.port());

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    log::info("shutting down");
    server.stop();
    engine.stop();
    return 0;
}

}  // namespace ember
