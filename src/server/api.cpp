#include "server/api.hpp"

#include <atomic>
#include <format>
#include <functional>
#include <random>
#include <thread>

#include "common/error.hpp"
#include "common/json.hpp"
#include "common/log.hpp"
#include "model/chat_template.hpp"
#include "server/text_stream.hpp"

namespace ember {

namespace web {
extern const unsigned char dashboard_html[];
extern const size_t dashboard_html_size;
}  // namespace web

namespace {

using json::Value;
using Clock = std::chrono::steady_clock;

// A client error: reported as 4xx in the API's own error format.
struct ApiError {
    int status;
    std::string message;
};

std::string make_id(const char *prefix) {
    static std::atomic<uint64_t> counter{0};
    thread_local std::mt19937_64 rng(std::random_device{}() ^ counter++);
    return std::format("{}{:016x}{:08x}", prefix, rng(), static_cast<uint32_t>(counter++));
}

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

Value parse_body(const http::Request &req) {
    try {
        Value v = Value::parse(req.body);
        if (!v.is_object()) throw ApiError{400, "the request body must be a JSON object"};
        return v;
    } catch (const Error &e) {
        throw ApiError{400, e.what()};
    }
}

// "content": a string, or an array of parts of which the text ones count.
std::string content_text(const Value *v) {
    if (!v || v->is_null()) return {};
    if (v->is_string()) return v->as_string();
    if (!v->is_array()) throw ApiError{400, "message content must be a string or an array of parts"};
    std::string out;
    for (const auto &part : v->as_array()) {
        if (part.is_string()) {
            out += part.as_string();
            continue;
        }
        std::string type = part.get_string("type", "text");
        if (type == "text" || type == "input_text") out += part.get_string("text", "");
        else throw ApiError{400, std::format("content parts of type \"{}\" are not supported (text only)", type)};
    }
    return out;
}

std::vector<std::string> stop_list(const Value *v) {
    std::vector<std::string> out;
    if (!v || v->is_null()) return out;
    if (v->is_string()) out.push_back(v->as_string());
    else
        for (const auto &s : v->as_array()) out.push_back(s.as_string());
    if (out.size() > 16) throw ApiError{400, "at most 16 stop sequences"};
    return out;
}

// What to generate, parsed from either API.
struct GenSpec {
    std::vector<ChatMessage> messages;
    bool raw = false;
    std::string raw_prompt;
    bool thinking = false;
    SamplingParams sampling;
    int max_tokens = 0;
    std::vector<std::string> stop;
    std::string tag;
};

struct GenResult {
    std::string text, thinking;
    FinishReason reason = FinishReason::none;
    bool stopped_by_string = false;
    std::string stop_matched;
    int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
    double first_token_ms = 0;
    bool client_gone = false;
};

void read_sampling(const Value &body, const ApiContext &ctx, GenSpec &spec) {
    spec.sampling = ctx.default_sampling;
    spec.sampling.temperature = static_cast<float>(body.get_number("temperature", spec.sampling.temperature));
    spec.sampling.top_p = static_cast<float>(body.get_number("top_p", spec.sampling.top_p));
    spec.sampling.top_k = static_cast<int>(body.get_int("top_k", spec.sampling.top_k));
    spec.sampling.min_p = static_cast<float>(body.get_number("min_p", spec.sampling.min_p));
    spec.sampling.seed = static_cast<uint64_t>(body.get_int("seed", 0));
    if (spec.sampling.temperature < 0 || spec.sampling.temperature > 5) throw ApiError{400, "temperature must be in [0, 5]"};
    if (spec.sampling.top_p <= 0 || spec.sampling.top_p > 1) throw ApiError{400, "top_p must be in (0, 1]"};
    if (spec.sampling.top_k < 0) throw ApiError{400, "top_k must be >= 0"};
    if (spec.sampling.min_p < 0 || spec.sampling.min_p >= 1) throw ApiError{400, "min_p must be in [0, 1)"};
}

std::vector<int32_t> prompt_ids(const ApiContext &ctx, const GenSpec &spec) {
    std::string text;
    if (spec.raw) {
        text = spec.raw_prompt;
    } else {
        if (spec.messages.empty()) throw ApiError{400, "messages must not be empty"};
        ChatOptions co;
        co.enable_thinking = spec.thinking;
        text = render_chat(spec.messages, co);
    }
    auto ids = ctx.tokenizer.encode(text);
    if (ids.empty()) throw ApiError{400, "the prompt is empty"};
    return {ids.begin(), ids.end()};
}

// Runs one generation, calling on_piece for streamed text (if given). Cancels
// the request when the client disconnects.
GenResult run(ApiContext &ctx, const GenSpec &spec, std::vector<int32_t> ids, http::Response &res,
              const std::function<bool(const TextStream::Piece &)> &on_piece) {
    GenResult out;
    out.prompt_tokens = static_cast<int>(ids.size());
    Request r;
    r.prompt = std::move(ids);
    r.sampling = spec.sampling;
    r.max_tokens = spec.max_tokens;
    r.tag = spec.tag;
    std::shared_ptr<RequestStream> stream;
    try {
        stream = ctx.engine.submit(std::move(r));
    } catch (const Error &e) {
        throw ApiError{400, e.what()};
    }
    TextStream ts(ctx.tokenizer, spec.stop, false);
    auto deliver = [&](const TextStream::Piece &p) {
        out.text += p.text;
        out.thinking += p.thinking;
        if (on_piece && (!p.text.empty() || !p.thinking.empty()) && !on_piece(p)) {
            out.client_gone = true;
            return false;
        }
        return true;
    };
    Delta d;
    for (;;) {
        if (!stream->next(d, std::chrono::milliseconds(200))) {
            if (res.client_gone()) {
                out.client_gone = true;
                stream->cancel();
                break;
            }
            continue;
        }
        out.completion_tokens = d.generated;
        out.cached_tokens = d.cached_tokens;
        if (d.first_token_ms > 0) out.first_token_ms = d.first_token_ms;
        if (!deliver(ts.push(d.tokens))) {
            stream->cancel();
            break;
        }
        if (ts.stopped()) {
            stream->cancel();
            out.reason = FinishReason::stop;
            out.stopped_by_string = true;
            out.stop_matched = ts.stop_matched();
            break;
        }
        if (d.finished) {
            out.reason = d.reason;
            break;
        }
    }
    if (!out.client_gone) deliver(ts.flush());
    // Trim the blank lines Qwen3 puts around its reasoning.
    while (!out.thinking.empty() && (out.thinking.back() == '\n' || out.thinking.back() == ' ')) out.thinking.pop_back();
    size_t lead = out.thinking.find_first_not_of('\n');
    out.thinking.erase(0, lead == std::string::npos ? out.thinking.size() : lead);
    return out;
}

std::string sse(const Value &v) { return "data: " + v.dump() + "\n\n"; }
std::string sse_event(const char *event, const Value &v) { return std::string("event: ") + event + "\ndata: " + v.dump() + "\n\n"; }

// ---- OpenAI --------------------------------------------------------------------------

void openai_error(http::Response &res, int status, const std::string &message) {
    Value e;
    e.set("message", message);
    e.set("type", status == 400 ? "invalid_request_error" : "server_error");
    Value body;
    body.set("error", std::move(e));
    res.respond(status, "application/json", body.dump());
}

const char *openai_finish(const GenResult &r) {
    switch (r.reason) {
        case FinishReason::length: return "length";
        case FinishReason::error: return "error";
        default: return "stop";
    }
}

Value openai_usage(const GenResult &r) {
    Value u;
    u.set("prompt_tokens", r.prompt_tokens);
    u.set("completion_tokens", r.completion_tokens);
    u.set("total_tokens", r.prompt_tokens + r.completion_tokens);
    Value details;
    details.set("cached_tokens", r.cached_tokens);
    u.set("prompt_tokens_details", std::move(details));
    return u;
}

void chat_completions(ApiContext &ctx, const http::Request &req, http::Response &res, bool legacy) {
    Value body = parse_body(req);
    GenSpec spec;
    spec.tag = legacy ? "openai completions" : "openai chat";
    if (legacy) {
        spec.raw = true;
        const Value &p = body["prompt"];
        spec.raw_prompt = p.is_array() ? p[0].as_string() : p.as_string();
    } else {
        const Value *msgs = body.find("messages");
        if (!msgs || !msgs->is_array()) throw ApiError{400, "messages is required and must be an array"};
        for (const auto &m : msgs->as_array()) {
            std::string role = m.get_string("role", "");
            if (role == "developer") role = "system";
            if (role != "system" && role != "user" && role != "assistant" && role != "tool")
                throw ApiError{400, std::format("unsupported message role \"{}\"", role)};
            spec.messages.push_back({role, content_text(m.find("content"))});
        }
        bool think = body.get_bool("enable_thinking", false);
        if (const Value *kw = body.find("chat_template_kwargs"); kw && kw->is_object()) think = kw->get_bool("enable_thinking", think);
        spec.thinking = think;
    }
    if (body.get_int("n", 1) != 1) throw ApiError{400, "only n = 1 is supported"};
    spec.max_tokens = static_cast<int>(body.get_int("max_completion_tokens", body.get_int("max_tokens", ctx.default_max_tokens)));
    if (spec.max_tokens <= 0) throw ApiError{400, "max_tokens must be positive"};
    read_sampling(body, ctx, spec);
    spec.stop = stop_list(body.find("stop"));
    const bool stream = body.get_bool("stream", false);
    bool usage_in_stream = false;
    if (const Value *so = body.find("stream_options"); so && so->is_object()) usage_in_stream = so->get_bool("include_usage", false);

    auto ids = prompt_ids(ctx, spec);
    const std::string id = make_id(legacy ? "cmpl-" : "chatcmpl-");
    const int64_t created = unix_now();
    const char *object = legacy ? "text_completion" : "chat.completion";

    if (!stream) {
        GenResult r = run(ctx, spec, std::move(ids), res, nullptr);
        if (r.client_gone) return;
        Value choice;
        choice.set("index", 0);
        if (legacy) {
            choice.set("text", r.text);
        } else {
            Value msg;
            msg.set("role", "assistant");
            msg.set("content", r.text);
            if (!r.thinking.empty()) msg.set("reasoning_content", r.thinking);
            choice.set("message", std::move(msg));
        }
        choice.set("finish_reason", openai_finish(r));
        Value out;
        out.set("id", id);
        out.set("object", object);
        out.set("created", created);
        out.set("model", ctx.model_name);
        out.set("choices", json::Array{std::move(choice)});
        out.set("usage", openai_usage(r));
        res.respond(200, "application/json", out.dump());
        return;
    }

    res.start_stream(200, "text/event-stream");
    auto chunk = [&](Value delta_or_text, const char *finish) {
        Value choice;
        choice.set("index", 0);
        if (legacy) choice.set("text", std::move(delta_or_text));
        else choice.set("delta", std::move(delta_or_text));
        choice.set("finish_reason", finish ? Value(finish) : Value());
        Value c;
        c.set("id", id);
        c.set("object", legacy ? "text_completion" : "chat.completion.chunk");
        c.set("created", created);
        c.set("model", ctx.model_name);
        c.set("choices", json::Array{std::move(choice)});
        return c;
    };
    if (!legacy) {
        Value role;
        role.set("role", "assistant");
        role.set("content", "");
        if (!res.write(sse(chunk(std::move(role), nullptr)))) return;
    }
    GenResult r = run(ctx, spec, std::move(ids), res, [&](const TextStream::Piece &p) {
        if (legacy) return res.write(sse(chunk(Value(p.text), nullptr)));
        Value delta;
        if (!p.thinking.empty()) delta.set("reasoning_content", p.thinking);
        if (!p.text.empty()) delta.set("content", p.text);
        return res.write(sse(chunk(std::move(delta), nullptr)));
    });
    if (r.client_gone) return;
    res.write(sse(chunk(legacy ? Value("") : Value(json::Object{}), openai_finish(r))));
    if (usage_in_stream) {
        Value u;
        u.set("id", id);
        u.set("object", legacy ? "text_completion" : "chat.completion.chunk");
        u.set("created", created);
        u.set("model", ctx.model_name);
        u.set("choices", json::Array{});
        u.set("usage", openai_usage(r));
        res.write(sse(u));
    }
    res.write("data: [DONE]\n\n");
    res.finish();
}

// ---- Anthropic -----------------------------------------------------------------------

void anthropic_error(http::Response &res, int status, const std::string &message) {
    Value e;
    e.set("type", status == 400 ? "invalid_request_error" : "api_error");
    e.set("message", message);
    Value body;
    body.set("type", "error");
    body.set("error", std::move(e));
    res.respond(status, "application/json", body.dump());
}

const char *anthropic_stop(const GenResult &r) {
    if (r.stopped_by_string) return "stop_sequence";
    if (r.reason == FinishReason::length) return "max_tokens";
    return "end_turn";
}

Value anthropic_usage(const GenResult &r, bool final_counts) {
    Value u;
    u.set("input_tokens", r.prompt_tokens - r.cached_tokens);
    u.set("cache_read_input_tokens", r.cached_tokens);
    u.set("cache_creation_input_tokens", 0);
    u.set("output_tokens", final_counts ? r.completion_tokens : 1);
    return u;
}

void messages(ApiContext &ctx, const http::Request &req, http::Response &res) {
    Value body = parse_body(req);
    GenSpec spec;
    spec.tag = "anthropic messages";
    const Value *mt = body.find("max_tokens");
    if (!mt || !mt->is_number()) throw ApiError{400, "max_tokens: field required"};
    spec.max_tokens = static_cast<int>(mt->as_int());
    if (spec.max_tokens <= 0) throw ApiError{400, "max_tokens must be positive"};
    if (const Value *sys = body.find("system"); sys && !sys->is_null()) spec.messages.push_back({"system", content_text(sys)});
    const Value *msgs = body.find("messages");
    if (!msgs || !msgs->is_array() || msgs->size() == 0) throw ApiError{400, "messages: field required"};
    for (const auto &m : msgs->as_array()) {
        std::string role = m.get_string("role", "");
        if (role != "user" && role != "assistant") throw ApiError{400, std::format("unsupported message role \"{}\"", role)};
        spec.messages.push_back({role, content_text(m.find("content"))});
    }
    if (const Value *th = body.find("thinking"); th && th->is_object()) spec.thinking = th->get_string("type", "") == "enabled";
    read_sampling(body, ctx, spec);
    spec.stop = stop_list(body.find("stop_sequences"));
    const bool stream = body.get_bool("stream", false);

    auto ids = prompt_ids(ctx, spec);
    const std::string id = make_id("msg_");
    auto message_shell = [&](const GenResult *r) {
        Value m;
        m.set("id", id);
        m.set("type", "message");
        m.set("role", "assistant");
        m.set("model", ctx.model_name);
        json::Array content;
        if (r) {
            if (!r->thinking.empty()) {
                Value t;
                t.set("type", "thinking");
                t.set("thinking", r->thinking);
                t.set("signature", "");
                content.push_back(std::move(t));
            }
            Value t;
            t.set("type", "text");
            t.set("text", r->text);
            content.push_back(std::move(t));
        }
        m.set("content", std::move(content));
        m.set("stop_reason", r ? Value(anthropic_stop(*r)) : Value());
        m.set("stop_sequence", r && r->stopped_by_string ? Value(r->stop_matched) : Value());
        return m;
    };

    if (!stream) {
        GenResult r = run(ctx, spec, std::move(ids), res, nullptr);
        if (r.client_gone) return;
        Value m = message_shell(&r);
        m.set("usage", anthropic_usage(r, true));
        res.respond(200, "application/json", m.dump());
        return;
    }

    res.start_stream(200, "text/event-stream");
    {
        GenResult pre;
        pre.prompt_tokens = static_cast<int>(ids.size());
        Value start;
        start.set("type", "message_start");
        Value m = message_shell(nullptr);
        m.set("usage", anthropic_usage(pre, false));
        start.set("message", std::move(m));
        if (!res.write(sse_event("message_start", start))) return;
    }
    // Content blocks open lazily: a thinking block (if the model reasons), then the text block.
    int index = -1;
    std::string open_type;
    auto open_block = [&](const char *type) {
        if (open_type == type) return true;
        if (!open_type.empty()) {
            Value stop;
            stop.set("type", "content_block_stop");
            stop.set("index", index);
            if (!res.write(sse_event("content_block_stop", stop))) return false;
        }
        index++;
        open_type = type;
        Value block;
        block.set("type", type);
        if (open_type == "thinking") {
            block.set("thinking", "");
            block.set("signature", "");
        } else {
            block.set("text", "");
        }
        Value start;
        start.set("type", "content_block_start");
        start.set("index", index);
        start.set("content_block", std::move(block));
        return res.write(sse_event("content_block_start", start));
    };
    auto delta = [&](const char *type, const char *field, const std::string &text, const char *block) {
        if (!open_block(block)) return false;
        Value d;
        d.set("type", type);
        d.set(field, text);
        Value ev;
        ev.set("type", "content_block_delta");
        ev.set("index", index);
        ev.set("delta", std::move(d));
        return res.write(sse_event("content_block_delta", ev));
    };
    GenResult r = run(ctx, spec, std::move(ids), res, [&](const TextStream::Piece &p) {
        if (!p.thinking.empty() && !delta("thinking_delta", "thinking", p.thinking, "thinking")) return false;
        if (!p.text.empty() && !delta("text_delta", "text", p.text, "text")) return false;
        return true;
    });
    if (r.client_gone) return;
    if (open_type.empty()) open_block("text");  // always at least one (possibly empty) text block
    Value stop;
    stop.set("type", "content_block_stop");
    stop.set("index", index);
    res.write(sse_event("content_block_stop", stop));
    Value md;
    md.set("type", "message_delta");
    Value d;
    d.set("stop_reason", anthropic_stop(r));
    d.set("stop_sequence", r.stopped_by_string ? Value(r.stop_matched) : Value());
    md.set("delta", std::move(d));
    md.set("usage", anthropic_usage(r, true));
    res.write(sse_event("message_delta", md));
    Value ms;
    ms.set("type", "message_stop");
    res.write(sse_event("message_stop", ms));
    res.finish();
}

// ---- monitoring --------------------------------------------------------------------------

Value stats_json(ApiContext &ctx) {
    EngineStats s = ctx.engine.stats();
    MemoryInfo mem = ctx.engine.backend().memory();
    Value v;
    v.set("model", ctx.model_name);
    v.set("backend", ctx.backend_name);
    v.set("weights", ctx.weights);
    v.set("uptime_s", s.uptime_s);
    v.set("steps", static_cast<int64_t>(s.steps));
    v.set("requests_done", static_cast<int64_t>(s.requests_done));
    v.set("prompt_tokens", static_cast<int64_t>(s.prompt_tokens));
    v.set("cached_tokens", static_cast<int64_t>(s.cached_tokens));
    v.set("generated_tokens", static_cast<int64_t>(s.generated_tokens));
    v.set("preemptions", static_cast<int64_t>(s.preemptions));
    v.set("running", s.running);
    v.set("waiting", s.waiting);
    v.set("tokens_per_second", s.tokens_per_second);
    v.set("prefill_per_second", s.prefill_per_second);
    v.set("step_ms", s.step_ms);
    v.set("batch_tokens", s.last_batch_tokens);
    v.set("batch_seqs", s.last_batch_seqs);
    v.set("ttft_ms", s.ttft_ms_avg);
    Value kv;
    kv.set("blocks", s.kv_blocks);
    kv.set("used", s.kv_used);
    kv.set("cached", s.kv_cached);
    kv.set("block_size", kBlockSize);
    std::string map;
    for (uint8_t b : ctx.engine.kv_block_states()) map += static_cast<char>('0' + b);
    kv.set("map", map);
    v.set("kv", std::move(kv));
    Value m;
    m.set("weights_mib", static_cast<double>(mem.weights) / (1 << 20));
    m.set("kv_cache_mib", static_cast<double>(mem.kv_cache) / (1 << 20));
    m.set("buffers_mib", static_cast<double>(mem.activations) / (1 << 20));
    m.set("device_total_mib", static_cast<double>(mem.device_total) / (1 << 20));
    m.set("device_free_mib", static_cast<double>(mem.device_free) / (1 << 20));
    v.set("memory", std::move(m));
    json::Array reqs;
    for (const auto &r : ctx.engine.requests()) {
        Value q;
        q.set("id", static_cast<int64_t>(r.id));
        q.set("tag", r.tag);
        q.set("state", r.state);
        q.set("prompt", r.prompt_tokens);
        q.set("cached", r.cached_tokens);
        q.set("generated", r.generated);
        q.set("max_tokens", r.max_tokens);
        q.set("blocks", r.blocks);
        q.set("age_ms", r.age_ms);
        reqs.push_back(std::move(q));
    }
    v.set("requests", std::move(reqs));
    return v;
}

std::string prometheus(ApiContext &ctx) {
    EngineStats s = ctx.engine.stats();
    std::string out;
    auto metric = [&](const char *name, const char *type, const char *help, double value) {
        out += std::format("# HELP {} {}\n# TYPE {} {}\n{} {}\n", name, help, name, type, name, value);
    };
    metric("ember_generated_tokens_total", "counter", "Tokens generated.", static_cast<double>(s.generated_tokens));
    metric("ember_prompt_tokens_total", "counter", "Prompt tokens computed.", static_cast<double>(s.prompt_tokens));
    metric("ember_cached_prompt_tokens_total", "counter", "Prompt tokens served from the prefix cache.",
           static_cast<double>(s.cached_tokens));
    metric("ember_requests_total", "counter", "Requests completed.", static_cast<double>(s.requests_done));
    metric("ember_preemptions_total", "counter", "Sequences preempted for lack of KV cache.", static_cast<double>(s.preemptions));
    metric("ember_steps_total", "counter", "Engine steps.", static_cast<double>(s.steps));
    metric("ember_requests_running", "gauge", "Requests being processed.", s.running);
    metric("ember_requests_waiting", "gauge", "Requests queued.", s.waiting);
    metric("ember_kv_blocks_total", "gauge", "KV cache blocks.", s.kv_blocks);
    metric("ember_kv_blocks_used", "gauge", "KV cache blocks in use.", s.kv_used);
    metric("ember_generation_tokens_per_second", "gauge", "Recent generation throughput.", s.tokens_per_second);
    metric("ember_time_to_first_token_ms", "gauge", "Recent average time to first token.", s.ttft_ms_avg);
    return out;
}

template <class F>
http::Handler guarded(ApiContext &ctx, F f, bool anthropic) {
    return [&ctx, f, anthropic](const http::Request &req, http::Response &res) {
        try {
            f(ctx, req, res);
        } catch (const ApiError &e) {
            if (res.sent()) {
                res.finish();
                return;
            }
            if (anthropic) anthropic_error(res, e.status, e.message);
            else openai_error(res, e.status, e.message);
        } catch (const Error &e) {
            if (res.sent()) {
                res.finish();
                return;
            }
            if (anthropic) anthropic_error(res, 400, e.what());
            else openai_error(res, 400, e.what());
        }
    };
}

}  // namespace

void register_api(http::Server &server, ApiContext &ctx) {
    server.route("POST", "/v1/chat/completions",
                 guarded(ctx, [](ApiContext &c, const http::Request &q, http::Response &r) { chat_completions(c, q, r, false); }, false));
    server.route("POST", "/v1/completions",
                 guarded(ctx, [](ApiContext &c, const http::Request &q, http::Response &r) { chat_completions(c, q, r, true); }, false));
    server.route("POST", "/v1/messages", guarded(ctx, [](ApiContext &c, const http::Request &q, http::Response &r) { messages(c, q, r); }, true));
    server.route("GET", "/v1/models", [&ctx](const http::Request &, http::Response &res) {
        Value m;
        m.set("id", ctx.model_name);
        m.set("object", "model");
        m.set("created", unix_now());
        m.set("owned_by", "ember");
        Value out;
        out.set("object", "list");
        out.set("data", json::Array{std::move(m)});
        res.respond(200, "application/json", out.dump());
    });
    server.route("GET", "/health", [](const http::Request &, http::Response &res) {
        res.respond(200, "application/json", R"({"status":"ok"})");
    });
    server.route("GET", "/metrics", [&ctx](const http::Request &, http::Response &res) {
        res.respond(200, "text/plain; version=0.0.4", prometheus(ctx));
    });
    server.route("GET", "/api/stats", [&ctx](const http::Request &, http::Response &res) {
        res.respond(200, "application/json", stats_json(ctx).dump());
    });
    server.route("GET", "/api/events", [&ctx](const http::Request &, http::Response &res) {
        if (!res.start_stream(200, "text/event-stream")) return;
        while (res.write(sse(stats_json(ctx)))) {
            for (int i = 0; i < 5; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (res.client_gone()) return;
            }
        }
    });
    auto dashboard = [](const http::Request &, http::Response &res) {
        res.respond(200, "text/html; charset=utf-8",
                    std::string(reinterpret_cast<const char *>(web::dashboard_html), web::dashboard_html_size));
    };
    server.route("GET", "/", dashboard);
    server.route("GET", "/dashboard", dashboard);
}

}  // namespace ember
