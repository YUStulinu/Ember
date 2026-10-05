// The HTTP server and both APIs, end to end: a real server on a free port, the
// engine on the fake backend (deterministic output), and real HTTP requests.
#include <chrono>
#include <cstring>
#include <string>

#include "common/json.hpp"
#include "engine/engine.hpp"
#include "fake_backend.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"
#include "server/api.hpp"
#include "server/http.hpp"
#include "test.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace ember;
using namespace ember::test;

namespace {

struct HttpResult {
    int status = 0;
    std::string headers, body;
};

// One request with "Connection: close"; reads everything; de-chunks the body.
HttpResult http_call(int port, const std::string &method, const std::string &path, const std::string &body = "") {
#ifdef _WIN32
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(s, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0) throw Failure{"cannot connect to the test server"};
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n";
    if (!body.empty()) req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    send(s, req.data(), static_cast<int>(req.size()), 0);
    std::string raw;
    char buf[8192];
    for (;;) {
        int got = recv(s, buf, sizeof buf, 0);
        if (got <= 0) break;
        raw.append(buf, static_cast<size_t>(got));
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    HttpResult r;
    size_t head_end = raw.find("\r\n\r\n");
    if (head_end == std::string::npos) throw Failure{"malformed HTTP response"};
    r.headers = raw.substr(0, head_end);
    r.status = std::atoi(raw.c_str() + raw.find(' ') + 1);
    std::string rest = raw.substr(head_end + 4);
    if (r.headers.find("Transfer-Encoding: chunked") != std::string::npos) {
        size_t pos = 0;
        while (pos < rest.size()) {
            size_t eol = rest.find("\r\n", pos);
            size_t n = std::stoul(rest.substr(pos, eol - pos), nullptr, 16);
            if (n == 0) break;
            r.body += rest.substr(eol + 2, n);
            pos = eol + 2 + n + 2;
        }
    } else {
        r.body = rest;
    }
    return r;
}

// Splits an SSE body into (event, data) pairs.
std::vector<std::pair<std::string, std::string>> sse_events(const std::string &body) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t pos = 0;
    while (pos < body.size()) {
        size_t end = body.find("\n\n", pos);
        if (end == std::string::npos) break;
        std::string block = body.substr(pos, end - pos);
        pos = end + 2;
        std::string event, data;
        size_t lp = 0;
        while (lp < block.size()) {
            size_t le = block.find('\n', lp);
            if (le == std::string::npos) le = block.size();
            std::string line = block.substr(lp, le - lp);
            if (line.rfind("event: ", 0) == 0) event = line.substr(7);
            if (line.rfind("data: ", 0) == 0) data = line.substr(6);
            lp = le + 1;
        }
        out.emplace_back(event, data);
    }
    return out;
}

struct TestServer {
    const Tokenizer &tok;
    FakeBackend backend;
    Engine engine;
    ApiContext ctx;
    http::Server server;

    explicit TestServer(const Tokenizer &t)
        : tok(t), backend(512, 512, 16, false, 0, 0, 151936), engine(backend, EngineOptions{}), ctx(engine, tok) {
        ctx.model_name = "fake";
        ctx.default_sampling.temperature = 0;
        engine.start();
        register_api(server, ctx);
        server.start("127.0.0.1", 0);
    }
    ~TestServer() {
        server.stop();
        engine.stop();
    }
};

const Tokenizer &tokenizer() {
    require_tokenizer("Qwen3-0.6B");
    static const Tokenizer tok = Tokenizer::load(model_dir("Qwen3-0.6B") + "/tokenizer.json");
    return tok;
}

// What the fake model answers to a chat, decoded as the API would.
std::string expected_answer(const Tokenizer &tok, const std::vector<ChatMessage> &msgs, int max_tokens) {
    auto ids = tok.encode(render_chat(msgs, ChatOptions{}));
    auto out = expected(std::vector<int32_t>(ids.begin(), ids.end()), max_tokens, false);
    std::string text = tok.decode(std::vector<int>(out.begin(), out.end()), true);
    size_t first = text.find_first_not_of('\n');
    return first == std::string::npos ? std::string() : text.substr(first);
}

}  // namespace

TEST("server: OpenAI chat completions, plain and streaming") {
    const Tokenizer &tok = tokenizer();
    TestServer ts(tok);
    const int port = ts.server.port();
    const std::string want = expected_answer(tok, {{"user", "Hello there"}}, 24);

    auto r = http_call(port, "POST", "/v1/chat/completions",
                       R"({"model":"x","messages":[{"role":"user","content":"Hello there"}],"max_tokens":24})");
    CHECK_EQ(r.status, 200);
    auto j = json::Value::parse(r.body);
    CHECK_EQ(j["object"].as_string(), std::string("chat.completion"));
    CHECK_EQ(j["choices"][0]["message"]["content"].as_string(), want);
    CHECK_EQ(j["choices"][0]["finish_reason"].as_string(), std::string("length"));
    CHECK_EQ(j["usage"]["completion_tokens"].as_int(), 24);

    r = http_call(port, "POST", "/v1/chat/completions",
                  R"({"messages":[{"role":"user","content":"Hello there"}],"max_tokens":24,"stream":true,"stream_options":{"include_usage":true}})");
    CHECK_EQ(r.status, 200);
    CHECK(r.headers.find("text/event-stream") != std::string::npos);
    std::string streamed;
    bool done = false, usage = false;
    for (const auto &[ev, data] : sse_events(r.body)) {
        if (data == "[DONE]") {
            done = true;
            continue;
        }
        auto c = json::Value::parse(data);
        if (c.contains("usage")) usage = c["usage"]["completion_tokens"].as_int() == 24;
        if (c["choices"].size() && c["choices"][0]["delta"].contains("content"))
            streamed += c["choices"][0]["delta"]["content"].as_string();
    }
    CHECK(done && usage);
    CHECK_EQ(streamed, want);
}

TEST("server: Anthropic messages, plain and streaming") {
    const Tokenizer &tok = tokenizer();
    TestServer ts(tok);
    const int port = ts.server.port();
    const std::string want = expected_answer(tok, {{"system", "Be brief."}, {"user", "Hi"}}, 16);

    auto r = http_call(port, "POST", "/v1/messages",
                       R"({"model":"x","max_tokens":16,"system":"Be brief.","messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]})");
    CHECK_EQ(r.status, 200);
    auto j = json::Value::parse(r.body);
    CHECK_EQ(j["type"].as_string(), std::string("message"));
    CHECK_EQ(j["content"][0]["text"].as_string(), want);
    CHECK_EQ(j["stop_reason"].as_string(), std::string("max_tokens"));
    CHECK_EQ(j["usage"]["output_tokens"].as_int(), 16);

    r = http_call(port, "POST", "/v1/messages",
                  R"({"max_tokens":16,"stream":true,"system":"Be brief.","messages":[{"role":"user","content":"Hi"}]})");
    std::vector<std::string> types;
    std::string text;
    for (const auto &[ev, data] : sse_events(r.body)) {
        types.push_back(ev);
        auto e = json::Value::parse(data);
        CHECK_EQ(e["type"].as_string(), ev);
        if (ev == "content_block_delta") text += e["delta"]["text"].as_string();
    }
    CHECK(!types.empty() && types.front() == "message_start" && types.back() == "message_stop");
    CHECK(std::find(types.begin(), types.end(), "message_delta") != types.end());
    CHECK_EQ(text, want);
}

TEST("server: errors, health, metrics, stats") {
    const Tokenizer &tok = tokenizer();
    TestServer ts(tok);
    const int port = ts.server.port();
    auto bad = http_call(port, "POST", "/v1/chat/completions", "{not json");
    CHECK_EQ(bad.status, 400);
    CHECK(json::Value::parse(bad.body)["error"].contains("message"));
    auto missing = http_call(port, "POST", "/v1/messages", R"({"messages":[{"role":"user","content":"x"}]})");
    CHECK_EQ(missing.status, 400);
    CHECK_EQ(json::Value::parse(missing.body)["type"].as_string(), std::string("error"));
    CHECK_EQ(http_call(port, "GET", "/nope").status, 404);
    CHECK_EQ(http_call(port, "GET", "/v1/chat/completions").status, 405);
    CHECK_EQ(http_call(port, "GET", "/health").status, 200);
    auto metrics = http_call(port, "GET", "/metrics");
    CHECK(metrics.body.find("ember_generated_tokens_total") != std::string::npos);
    auto stats = json::Value::parse(http_call(port, "GET", "/api/stats").body);
    CHECK_EQ(stats["model"].as_string(), std::string("fake"));
    CHECK_EQ(static_cast<int>(stats["kv"]["map"].as_string().size()), stats["kv"]["blocks"].as_int());
    auto page = http_call(port, "GET", "/");
    CHECK(page.status == 200 && page.body.find("<title>Ember</title>") != std::string::npos);
}
