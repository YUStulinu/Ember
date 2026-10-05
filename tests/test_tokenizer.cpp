// The tokenizer and chat template, compared with Hugging Face's on the cases in
// tests/golden (produced by reference/golden.py).
#include <chrono>

#include "common/json.hpp"
#include "common/mmap_file.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"
#include "model/unicode.hpp"
#include "test.hpp"

using namespace ember;

namespace {

const Tokenizer &qwen_tokenizer() {
    test::require_tokenizer("Qwen3-0.6B");
    static const Tokenizer tok = Tokenizer::load(test::model_dir("Qwen3-0.6B") + "/tokenizer.json");
    return tok;
}

json::Value golden(const std::string &file) {
    std::string path = test::source_dir() + "/tests/golden/" + file;
    if (!file_exists(path)) throw test::Skip{"run reference/golden.py to create " + file};
    return json::Value::parse(read_file(path));
}

std::string ids_str(const std::vector<int> &ids) {
    std::string s;
    for (int id : ids) s += std::format("{} ", id);
    return s;
}

}  // namespace

TEST("tokenizer: pre-tokenizer splits like the Qwen regex") {
    auto split = [](std::string_view s) {
        std::vector<std::string_view> p;
        Tokenizer::pre_tokenize(s, p);
        std::vector<std::string> out(p.begin(), p.end());
        return out;
    };
    using V = std::vector<std::string>;
    CHECK(split("Hello world") == (V{"Hello", " world"}));
    CHECK(split("I'm here, they'LL go") == (V{"I", "'m", " here", ",", " they", "'LL", " go"}));
    CHECK(split("x = 123;\n\n  y") == (V{"x", " =", " ", "1", "2", "3", ";\n\n", " ", " y"}));
    CHECK(split("a  \n b") == (V{"a", "  \n", " b"}));
    CHECK(split("end   ") == (V{"end", "   "}));
    CHECK(split("((ș))") == (V{"((", "ș", "))"}));
}

TEST("tokenizer: matches the official tokenizer on hard inputs") {
    const Tokenizer &tok = qwen_tokenizer();
    json::Value g = golden("tokenizer.json");
    int n = 0;
    for (const auto &c : g["cases"].as_array()) {
        const std::string &text = c["text"].as_string();
        std::vector<int> want;
        for (const auto &id : c["ids"].as_array()) want.push_back(static_cast<int>(id.as_int()));
        std::vector<int> got = tok.encode(text);
        CHECK_MSG(got == want, "text \"{}\"\n      got  {}\n      want {}", text, ids_str(got), ids_str(want));
        CHECK_MSG(tok.decode(got) == c["decoded"].as_string(), "decode of \"{}\"", text);
        n++;
    }
    CHECK(n >= 30);
}

TEST("tokenizer: special tokens, round trips, streaming decode") {
    const Tokenizer &tok = qwen_tokenizer();
    CHECK_EQ(tok.token_id("<|im_end|>").value(), 151645);
    CHECK(tok.is_special(151645) && !tok.is_special(151667));  // <think> is added but not special
    std::vector<int> ids = tok.encode("<|im_start|>x");
    CHECK_EQ(ids.front(), 151644);
    CHECK_EQ(tok.encode("<|im_start|>x", false).size() > 2, true);  // parse_special=false: plain text

    std::string text = "Știința și tehnologia în România 🇷🇴 — 東京 ✓";
    std::vector<int> t = tok.encode(text);
    CHECK_EQ(tok.decode(t), text);
    StreamDecoder dec(tok);
    std::string streamed;
    for (int id : t) {
        std::string piece = dec.push(id);
        size_t i = 0;  // every chunk is valid UTF-8 on its own
        while (i < piece.size()) CHECK(unicode::next_cp(piece, i) != unicode::kReplacement);
        streamed += piece;
    }
    streamed += dec.flush();
    CHECK_EQ(streamed, text);
}

TEST("tokenizer: speed") {
    const Tokenizer &tok = qwen_tokenizer();
    std::string text;
    for (int i = 0; i < 2000; i++) text += "Inference engines turn model weights into text, token by token; ";
    for (int i = 0; i < 200; i++) text += "Bună ziua! Motorul de inferență rulează pe placa video. ";
    auto t0 = std::chrono::steady_clock::now();
    auto ids = tok.encode(text);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (test::verbose()) std::printf("[%zu bytes -> %zu tokens in %.1f ms] ", text.size(), ids.size(), s * 1e3);
    CHECK(s < 1.0);
    CHECK_EQ(tok.decode(ids), text);
}

TEST("chat template: matches transformers' rendering") {
    json::Value g = golden("chat_template.json");
    for (const auto &c : g["cases"].as_array()) {
        std::vector<ChatMessage> msgs;
        for (const auto &m : c["messages"].as_array()) msgs.push_back({m["role"].as_string(), m["content"].as_string()});
        ChatOptions opt;
        opt.enable_thinking = c["enable_thinking"].as_bool();
        std::string got = render_chat(msgs, opt);
        CHECK_MSG(got == c["prompt"].as_string(), "\n--- got:\n{}\n--- want:\n{}", got, c["prompt"].as_string());
    }
}
