// JSON, fp16 conversions, the thread pool, UTF-8 and NFC.
#include <atomic>
#include <cstring>
#include <random>

#include "common/error.hpp"
#include "common/fp16.hpp"
#include "common/json.hpp"
#include "common/thread_pool.hpp"
#include "model/unicode.hpp"
#include "test.hpp"

using namespace ember;

TEST("json: parse, access, round trip") {
    auto v = json::Value::parse(R"({"a": 1, "b": [true, null, -2.5e3, "x\"yé😀"], "c": {"d": 9007199254740993}})");
    CHECK_EQ(v["a"].as_int(), 1);
    CHECK(v["b"][0].as_bool());
    CHECK(v["b"][1].is_null());
    CHECK_NEAR(v["b"][2].as_number(), -2500.0, 0);
    CHECK_EQ(v["b"][3].as_string(), std::string("x\"y\xC3\xA9\xF0\x9F\x98\x80"));
    CHECK_EQ(v["c"]["d"].as_int(), 9007199254740993LL);  // exact beyond 2^53
    CHECK(v.find("missing") == nullptr);
    CHECK_EQ(v.get_int("missing", 7), 7);
    auto again = json::Value::parse(v.dump());
    CHECK_EQ(again.dump(), v.dump());
}

TEST("json: errors carry a position") {
    try {
        json::Value::parse("{\n  \"a\": [1, 2,]\n}");
        CHECK(false);
    } catch (const Error &e) {
        CHECK_MSG(std::strstr(e.what(), "line 2"), "message: {}", e.what());
    }
    CHECK_THROWS(json::Value::parse("[1, 2"));
    CHECK_THROWS(json::Value::parse("{\"a\" 1}"));
    CHECK_THROWS(json::Value::parse("01x"));
    CHECK_THROWS(json::Value::parse(std::string(1000, '[')));  // nesting limit, not a stack overflow
    CHECK_THROWS(json::Value::parse("\"bad \\q escape\""));
    CHECK_THROWS(json::Value::parse("1").as_string());
}

TEST("json: builders and escaping") {
    json::Value v;
    v.set("text", "line\nbreak \"quoted\" \x01");
    v.set("n", 3);
    v.set("list", json::Array{1, 2.5, "s"});
    v.set("n", 4);  // replaces
    CHECK_EQ(v.dump(), std::string(R"({"text":"line\nbreak \"quoted\" \u0001","n":4,"list":[1,2.5,"s"]})"));
}

TEST("fp16: conversions round-trip and round to nearest even") {
    for (uint32_t h = 0; h < 0x10000; h++) {
        float f = fp16_to_float(static_cast<uint16_t>(h));
        if (std::isnan(f)) continue;
        CHECK_MSG(float_to_fp16(f) == h, "half 0x{:04X}", h);
    }
    CHECK_EQ(float_to_fp16(65504.0f), 0x7BFF);
    CHECK_EQ(float_to_fp16(65520.0f), 0x7C00);  // rounds up to infinity
    CHECK_EQ(float_to_fp16(1.0f + 1.0f / 2048), 0x3C00);  // tie -> even
    CHECK_EQ(float_to_fp16(5.96e-8f), 0x0001);  // smallest subnormal
    for (uint32_t h = 0; h < 0x10000; h += 7) {
        float f = bf16_to_float(static_cast<uint16_t>(h));
        if (!std::isnan(f)) CHECK_EQ(float_to_bf16(f), h);
    }
}

TEST("thread pool: covers every index exactly once") {
    ThreadPool pool(8);
    for (size_t n : {0u, 1u, 7u, 1000u, 123457u}) {
        std::vector<std::atomic<int>> hits(n);
        pool.parallel_for(n, 3, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; i++) hits[i]++;
        });
        for (size_t i = 0; i < n; i++) CHECK_EQ(hits[i].load(), 1);
    }
    std::atomic<long long> sum{0};
    for (int rep = 0; rep < 200; rep++)  // many small jobs back to back
        pool.parallel_for(64, 1, [&](size_t b, size_t e) { sum += static_cast<long long>(e - b); });
    CHECK_EQ(sum.load(), 200LL * 64);
}

TEST("unicode: UTF-8 decoding and repair") {
    std::string s = "a\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80";
    size_t i = 0;
    CHECK_EQ(unicode::next_cp(s, i), static_cast<uint32_t>('a'));
    CHECK_EQ(unicode::next_cp(s, i), 0xE9u);
    CHECK_EQ(unicode::next_cp(s, i), 0x20ACu);
    CHECK_EQ(unicode::next_cp(s, i), 0x1F600u);
    CHECK_EQ(i, s.size());
    // Maximal subparts: a truncated 3-byte sequence is one U+FFFD, a stray continuation byte another.
    CHECK_EQ(unicode::to_valid_utf8("x\xE2\x82y\x80z"), std::string("x\xEF\xBF\xBDy\xEF\xBF\xBDz"));
    CHECK_EQ(unicode::to_valid_utf8("\xED\xA0\x80"), std::string("\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"));  // surrogate
    CHECK_EQ(unicode::complete_utf8_prefix("ab\xC8"), 2u);         // first byte of "ș" held back
    CHECK_EQ(unicode::complete_utf8_prefix("ab\xC8\x99"), 4u);
    CHECK_EQ(unicode::complete_utf8_prefix("\xF0\x9F\x98"), 0u);
}

TEST("unicode: NFC composes, reorders and leaves NFC text alone") {
    CHECK_EQ(unicode::nfc("s\xCC\xA6"), std::string("\xC8\x99"));                  // s + comma below -> ș
    CHECK_EQ(unicode::nfc("e\xCC\x81"), std::string("\xC3\xA9"));                  // e + acute -> é
    CHECK_EQ(unicode::nfc("\xE2\x84\xAB"), std::string("\xC3\x85"));               // Angstrom sign -> Å
    CHECK_EQ(unicode::nfc("a\xCC\x9B\xCC\x80"), unicode::nfc("a\xCC\x80\xCC\x9B")); // marks reordered
    CHECK_EQ(unicode::nfc("\xE1\x84\x80\xE1\x85\xA1"), std::string("\xEA\xB0\x80"));  // Hangul L+V -> 가
    std::string plain = "Greetings from Iași, Brăila and Timișoara!";
    CHECK_EQ(unicode::nfc(plain), plain);
    CHECK(unicode::is_letter(0x0219) && unicode::is_letter('Q') && !unicode::is_letter('1'));
    CHECK(unicode::is_number(0x0663) && unicode::is_number(0x2167));
    CHECK(unicode::is_space(0x3000) && unicode::is_space('\t') && !unicode::is_space(0x200B));
}
