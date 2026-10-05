// ember: the command-line front end.
//
//   ember generate [-p TEXT]     one completion, streamed
//   ember chat                   interactive multi-turn chat
//   ember serve                  OpenAI/Anthropic-compatible HTTP server + dashboard
//   ember bench                  throughput benchmark
//   ember info                   model and memory plan
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "backend/backend.hpp"
#include "cli_args.hpp"
#include "loader.hpp"
#include "common/error.hpp"
#include "common/log.hpp"
#include "engine/engine.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"

#ifdef EMBER_WITH_CUDA
#include "backend/cuda/cuda_testing.hpp"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace ember {

int run_server(const cli::Options &opt);  // server/server_main.cpp

namespace {

using Clock = std::chrono::steady_clock;

double since_ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void print_help() {
    std::puts(
        "Ember - an LLM inference engine written from scratch\n"
        "\n"
        "usage: ember <command> [options]\n"
        "\n"
        "commands:\n"
        "  generate   complete one prompt and stream the answer      (-p TEXT)\n"
        "  chat       interactive chat; earlier turns stay in the KV cache\n"
        "  serve      HTTP server: /v1/chat/completions, /v1/messages, dashboard at /\n"
        "  bench      prefill, decode and batched throughput\n"
        "  info       model, memory plan and KV cache capacity\n"
        "\n"
        "model and hardware:\n"
        "  -m, --model DIR        Hugging Face model directory (default: models/Qwen3-1.7B)\n"
        "  --device cuda|cpu      (default: cuda when available)\n"
        "  -q, --weights FORMAT   f16 | int8 | int4 (weight-only quantization)\n"
        "  --draft DIR            draft model for speculative decoding (same tokenizer)\n"
        "  --spec-tokens N        tokens the draft proposes per step (default 4)\n"
        "  --gpu-memory F         share of GPU memory to use (default 0.92)\n"
        "  --kv-cache-mib N       explicit KV cache size\n"
        "  --max-seqs N           sequences per step (default 64)\n"
        "  --max-batch-tokens N   tokens per step (default 2048)\n"
        "  --prefill-chunk N      prompt tokens per sequence per step (default 512)\n"
        "  --no-prefix-cache      disable reuse of cached prompt prefixes\n"
        "  --no-cuda-graphs       launch decode kernels one by one\n"
        "\n"
        "generation:\n"
        "  -p, --prompt TEXT      the user message (generate)\n"
        "  -s, --system TEXT      a system prompt\n"
        "  -n, --max-tokens N     (default 512)\n"
        "  --temp T --top-p P --top-k K --min-p P --seed S --greedy\n"
        "  --think                let Qwen3 think before answering\n"
        "  --raw                  use the prompt as-is, without the chat template\n"
        "\n"
        "server:\n"
        "  --host ADDR --port N   (default 127.0.0.1:8080)\n");
}

std::vector<int32_t> build_prompt(const cli::Options &opt, const Tokenizer &tok, const std::vector<ChatMessage> &history) {
    std::string text;
    if (opt.raw) {
        text = history.empty() ? std::string() : history.back().content;
    } else {
        ChatOptions co;
        co.enable_thinking = opt.think;
        text = render_chat(history, co);
    }
    auto ids = tok.encode(text);
    return std::vector<int32_t>(ids.begin(), ids.end());
}

// Streams one request's text to stdout; returns the generated text.
std::string stream_answer(Engine &engine, const Tokenizer &tok, Request req, bool show_stats) {
    const int prompt_tokens = static_cast<int>(req.prompt.size());
    auto t0 = Clock::now();
    auto stream = engine.submit(std::move(req));
    StreamDecoder dec(tok);
    std::string text;
    Delta d;
    double first_ms = 0;
    int generated = 0, cached = 0;
    FinishReason reason = FinishReason::none;
    for (;;) {
        if (!stream->next(d, std::chrono::milliseconds(100))) continue;
        if (!d.tokens.empty() && first_ms == 0) first_ms = since_ms(t0);
        for (int32_t t : d.tokens) {
            std::string piece = dec.push(t);
            text += piece;
            std::fwrite(piece.data(), 1, piece.size(), stdout);
        }
        std::fflush(stdout);
        generated = d.generated;
        cached = d.cached_tokens;
        if (d.finished) {
            reason = d.reason;
            break;
        }
    }
    std::string tail = dec.flush();
    text += tail;
    std::fwrite(tail.data(), 1, tail.size(), stdout);
    std::puts("");
    if (show_stats) {
        double total = since_ms(t0);
        double decode_s = (total - first_ms) / 1000.0;
        std::fprintf(stderr, "\n[%d prompt tokens (%d cached) | first token %.0f ms | %d tokens in %.2f s | %.1f tokens/s | %s]\n",
                     prompt_tokens, cached, first_ms, generated, total / 1000.0,
                     generated > 1 && decode_s > 0 ? (generated - 1) / decode_s : 0.0, finish_reason_name(reason));
    }
    return text;
}

int cmd_generate(const cli::Options &opt) {
    std::string prompt = opt.prompt;
    if (prompt.empty() && !opt.rest.empty()) prompt = opt.rest[0];
    if (prompt.empty()) fail("give a prompt with -p \"...\"");
    cli::Loaded l = cli::load(opt);
    Engine engine(*l.backend, opt.engine, l.draft.get());
    engine.start();
    std::vector<ChatMessage> history;
    if (!opt.system.empty()) history.push_back({"system", opt.system});
    history.push_back({"user", prompt});
    Request req;
    req.prompt = build_prompt(opt, l.tokenizer, history);
    req.sampling = opt.sampling;
    req.max_tokens = opt.max_tokens;
    req.tag = "cli";
    stream_answer(engine, l.tokenizer, std::move(req), true);
    EngineStats st = engine.stats();
    if (st.spec_steps)
        std::fprintf(stderr, "[speculative: %llu steps, %.0f%% of draft tokens accepted, %.2f tokens per target pass]\n",
                     static_cast<unsigned long long>(st.spec_steps),
                     100.0 * static_cast<double>(st.spec_accepted) / static_cast<double>(std::max<uint64_t>(1, st.spec_proposed)),
                     static_cast<double>(st.generated_tokens) / static_cast<double>(st.spec_steps));
    return 0;
}

int cmd_chat(cli::Options opt) {
    cli::Loaded l = cli::load(opt);
    Engine engine(*l.backend, opt.engine, l.draft.get());
    engine.start();
    std::vector<ChatMessage> history;
    if (!opt.system.empty()) history.push_back({"system", opt.system});
    std::fprintf(stderr, "Ember chat with %s. Commands: /reset, /think on|off, /quit\n", opt.model.c_str());
    std::string line;
    for (;;) {
        std::fputs("\nyou> ", stdout);
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            history.clear();
            if (!opt.system.empty()) history.push_back({"system", opt.system});
            std::puts("(conversation cleared)");
            continue;
        }
        if (line.rfind("/think", 0) == 0) {
            opt.think = line.find("on") != std::string::npos;
            std::printf("(thinking %s)\n", opt.think ? "on" : "off");
            continue;
        }
        if (line.empty()) continue;
        history.push_back({"user", line});
        Request req;
        req.prompt = build_prompt(opt, l.tokenizer, history);
        req.sampling = opt.sampling;
        req.max_tokens = opt.max_tokens;
        req.tag = "chat";
        std::fputs("ember> ", stdout);
        std::string answer = stream_answer(engine, l.tokenizer, std::move(req), opt.verbose);
        history.push_back({"assistant", answer});
    }
    return 0;
}

int cmd_info(const cli::Options &opt) {
    cli::Loaded l = cli::load(opt);
    const ModelConfig &c = l.backend->config();
    MemoryInfo m = l.backend->memory();
    auto mib = [](int64_t b) { return static_cast<double>(b) / (1 << 20); };
    std::printf("model:      %s\n            %s\n", opt.model.c_str(), c.describe().c_str());
    std::printf("tokenizer:  %d tokens\n", l.tokenizer.vocab_size());
    std::printf("backend:    %s, weights %s\n", l.backend->name().c_str(), weight_format_name(opt.backend.weights));
    std::printf("memory:     weights %.0f MiB, KV cache %.0f MiB, buffers %.0f MiB", mib(m.weights), mib(m.kv_cache),
                mib(m.activations));
    if (m.device_total) std::printf(" (GPU: %.0f MiB free of %.0f)", mib(m.device_free), mib(m.device_total));
    std::printf("\nKV cache:   %d blocks x %d tokens = %d tokens (%.1f KiB per token)\n", l.backend->num_kv_blocks(), kBlockSize,
                l.backend->num_kv_blocks() * kBlockSize, static_cast<double>(c.kv_bytes_per_token(2)) / 1024);
    return 0;
}

// ---- perplexity ---------------------------------------------------------------------
//
// exp(mean negative log-likelihood) of each token given the ones before it,
// over non-overlapping windows. The measure of what quantization costs.
int cmd_perplexity(cli::Options opt) {
    if (opt.rest.empty()) fail("give one or more text files: ember perplexity -m DIR [-q int4] eval/en.txt");
    int window = 1024;
    std::vector<std::string> files;
    for (const auto &a : opt.rest) {
        if (a.rfind("window=", 0) == 0) window = std::atoi(a.c_str() + 7);
        else files.push_back(a);
    }
    opt.backend.max_logit_rows = 64;
    cli::Loaded l = cli::load(opt);
    Backend &be = *l.backend;
    const int V = be.config().vocab_size, chunk = 64;
    EMBER_CHECK(be.num_kv_blocks() * kBlockSize >= window, "the KV cache is smaller than the window");
    std::vector<float> logits(static_cast<size_t>(V));
    std::printf("perplexity of %s (%s weights, windows of %d tokens)\n", opt.model.c_str(), weight_format_name(opt.backend.weights),
                window);
    for (const auto &file : files) {
        auto ids = l.tokenizer.encode(read_file(file), false);
        double nll = 0;
        long long counted = 0;
        auto t0 = Clock::now();
        for (size_t w0 = 0; w0 + 1 < ids.size(); w0 += static_cast<size_t>(window)) {
            const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(window), ids.size() - w0));
            for (int c0 = 0; c0 < n; c0 += chunk) {
                const int m = std::min(chunk, n - c0);
                StepBatch b;
                b.clear();
                for (int i = 0; i < m; i++) {
                    b.tokens.push_back(ids[w0 + static_cast<size_t>(c0 + i)]);
                    b.positions.push_back(c0 + i);
                    b.logit_rows.push_back(i);
                }
                b.query_start.push_back(m);
                b.context_len.push_back(c0 + m);
                b.max_blocks = (c0 + m + kBlockSize - 1) / kBlockSize;
                for (int k = 0; k < b.max_blocks; k++) b.block_tables.push_back(k);
                be.forward(b);
                for (int i = 0; i < m; i++) {
                    const size_t target = w0 + static_cast<size_t>(c0 + i + 1);
                    if (c0 + i + 1 >= n || target >= ids.size()) break;
                    be.logits(i, logits);
                    double mx = *std::max_element(logits.begin(), logits.end()), sum = 0;
                    for (float x : logits) sum += std::exp(static_cast<double>(x) - mx);
                    nll += mx + std::log(sum) - logits[static_cast<size_t>(ids[target])];
                    counted++;
                }
            }
        }
        std::printf("  %-24s %8.3f   (%lld tokens, %.1f s)\n", file.c_str(), std::exp(nll / std::max(1LL, counted)), counted,
                    since_ms(t0) / 1000);
    }
    return 0;
}

// ---- bench ------------------------------------------------------------------------

std::vector<int32_t> random_prompt(int n, std::mt19937 &rng) {
    std::uniform_int_distribution<int32_t> d(1000, 100000);
    std::vector<int32_t> p(static_cast<size_t>(n));
    for (auto &t : p) t = d(rng);
    return p;
}

#ifdef EMBER_WITH_CUDA
// GEMM micro-benchmark on the shapes of the model: effective weight bandwidth per kernel.
int cmd_kernels(const cli::Options &opt) {
    ModelConfig c = ModelConfig::load(opt.model);
    struct Shape {
        const char *name;
        int N, K, ep;
    };
    const Shape shapes[] = {{"qkv", c.q_dim() + 2 * c.kv_dim(), c.hidden, 0},
                            {"o", c.hidden, c.q_dim(), 2},
                            {"gate_up", 2 * c.intermediate, c.hidden, 3},
                            {"down", c.hidden, c.intermediate, 2},
                            {"lm_head", c.vocab_size, c.hidden, 1}};
    const char *names[] = {"auto", "gemv", "tc16", "tc32", "tc64", "tc128", "gemv_tc"};
    std::vector<int> ms = {1, 2, 4, 8, 16, 32, 64};
    if (!opt.rest.empty()) {
        ms.clear();
        for (auto &r : opt.rest) ms.push_back(std::atoi(r.c_str()));
    }
    std::printf("GEMM bandwidth (GB/s of weights), %s\n", opt.model.c_str());
    for (const Shape &s : shapes) {
        std::printf("%-8s N=%-6d K=%-5d", s.name, s.N, s.K);
        for (int k = 0; k < 7; k++) std::printf(" %8s", names[k]);
        std::puts("");
        for (int M : ms) {
            std::printf("   M=%-3d               ", M);
            for (int k = 0; k < 7; k++) {
                bool ok = (k != 1 || M <= 8) && (k != 6 || (M <= 64 && s.K % 32 == 0));
                if (!ok) {
                    std::printf(" %8s", "-");
                    continue;
                }
                double us = cuda::bench_gemm(M, s.N, s.K, s.ep, k, k >= 2 && k <= 5, 20);
                std::printf(" %8.0f", static_cast<double>(s.N) * s.K * 2 / (us * 1e3));
            }
            std::puts("");
        }
    }
    return 0;
}
#endif

int cmd_bench(cli::Options opt) {
    int prompt_len = 512, gen = 128;
    std::vector<int> batches = {1, 4, 8, 16, 32};
    for (size_t i = 0; i < opt.rest.size(); i++) {
        const std::string &a = opt.rest[i];
        if (a.rfind("prompt=", 0) == 0) prompt_len = std::atoi(a.c_str() + 7);
        else if (a.rfind("gen=", 0) == 0) gen = std::atoi(a.c_str() + 4);
        else if (a.rfind("batch=", 0) == 0) {
            batches.clear();
            std::string list = a.substr(6);
            for (size_t p = 0; p < list.size();) {
                size_t q = list.find(',', p);
                batches.push_back(std::atoi(list.substr(p, q - p).c_str()));
                p = q == std::string::npos ? list.size() : q + 1;
            }
        }
    }
    opt.engine.prefix_caching = false;  // measure computation, not cache hits
    cli::Loaded l = cli::load(opt);
    Engine engine(*l.backend, opt.engine, l.draft.get());
    std::mt19937 rng(1234);
    SamplingParams greedy;
    greedy.temperature = 0;

    auto run = [&](int n_req, int plen, int ntok) {
        std::vector<std::shared_ptr<RequestStream>> streams;
        auto t0 = Clock::now();
        for (int r = 0; r < n_req; r++) {
            Request req;
            req.prompt = random_prompt(plen, rng);
            req.sampling = greedy;
            req.max_tokens = ntok;
            req.ignore_eos = true;
            req.tag = "bench";
            streams.push_back(engine.submit(std::move(req)));
        }
        int done = 0;
        double first_token = 0;
        while (done < n_req) {
            engine.step();
            for (auto &s : streams) {
                Delta d;
                while (s->next(d, std::chrono::milliseconds(0))) {
                    if (d.generated >= 1 && first_token == 0) first_token = since_ms(t0);
                    if (d.finished) done++;
                }
            }
        }
        return std::pair<double, double>(since_ms(t0), first_token);
    };

    std::printf("Ember benchmark: %s on %s (%s weights), median of 3 runs\n", opt.model.c_str(), l.backend->name().c_str(),
                weight_format_name(opt.backend.weights));
    // A laptop GPU changes clocks with temperature and power: warm up, then report medians.
    run(1, 64, 16);
    run(4, 64, 32);
    auto median3 = [&](auto f) {
        std::vector<std::pair<double, double>> v = {f(), f(), f()};
        std::sort(v.begin(), v.end());
        return v[1];
    };
    auto [prefill_ms, ttft] = median3([&] { return run(1, prompt_len, 1); });
    std::printf("  prefill, %d-token prompt          %8.0f tokens/s   (%.1f ms)\n", prompt_len, prompt_len / (prefill_ms / 1000),
                prefill_ms);
    std::vector<double> rates;
    for (int i = 0; i < 3; i++) {
        auto [ms, first] = run(1, 32, gen);
        rates.push_back((gen - 1) / ((ms - first) / 1000));
    }
    std::sort(rates.begin(), rates.end());
    std::printf("  decode, 1 sequence                %8.1f tokens/s   (%.2f ms per token)\n", rates[1], 1000.0 / rates[1]);
    for (int b : batches) {
        if (b <= 1) continue;
        auto [ms, first] = median3([&] { return run(b, 64, gen); });
        std::printf("  decode, %2d sequences x %d tokens %8.1f tokens/s   (first token after %.0f ms)\n", b, gen,
                    b * gen / (ms / 1000), first);
    }
    EngineStats st = engine.stats();
    std::printf("  (%llu steps, %llu preemptions)\n", static_cast<unsigned long long>(st.steps),
                static_cast<unsigned long long>(st.preemptions));
    return 0;
}

}  // namespace
}  // namespace ember

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    using namespace ember;
    if (argc < 2 || std::string(argv[1]) == "help" || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_help();
        return argc < 2 ? 1 : 0;
    }
    std::string cmd = argv[1];
    try {
        cli::Options opt = cli::parse(argc, argv, 2);
        log::set_level(opt.verbose ? log::Level::debug : log::Level::info);
        if (cmd == "generate" || cmd == "gen") return cmd_generate(opt);
        if (cmd == "chat") return cmd_chat(opt);
        if (cmd == "serve") return run_server(opt);
        if (cmd == "bench") return cmd_bench(opt);
        if (cmd == "info") return cmd_info(opt);
        if (cmd == "perplexity" || cmd == "ppl") return cmd_perplexity(opt);
#ifdef EMBER_WITH_CUDA
        if (cmd == "kernels") return cmd_kernels(opt);
#endif
        std::fprintf(stderr, "unknown command \"%s\" (see ember help)\n", cmd.c_str());
        return 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
