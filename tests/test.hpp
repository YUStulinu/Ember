// A minimal test framework: TEST(name) { CHECK(...); }
//
// Tests that need the downloaded models call require_model(), which skips the
// test (rather than failing it) when the model is absent, as in CI.
#pragma once

#include <cmath>
#include <cstdio>
#include <format>
#include <functional>
#include <string>

namespace ember::test {

struct Failure {
    std::string message;
};
struct Skip {
    std::string reason;
};

void add(const char *name, std::function<void()> fn);
std::string source_dir();
std::string model_dir(const std::string &name);  // <repo>/models/<name>
void require_model(const std::string &name);     // weights present; throws Skip when missing
void require_tokenizer(const std::string &name); // tokenizer.json present (enough for tokenizer tests)
bool has_cuda();
void require_cuda();
uint64_t seed();
bool verbose();

struct Registrar {
    Registrar(const char *name, std::function<void()> fn) { add(name, std::move(fn)); }
};

}  // namespace ember::test

#define EMBER_TEST_CONCAT2(a, b) a##b
#define EMBER_TEST_CONCAT(a, b) EMBER_TEST_CONCAT2(a, b)
#define TEST(name)                                                                                   \
    static void EMBER_TEST_CONCAT(test_fn_, __LINE__)();                                             \
    static ::ember::test::Registrar EMBER_TEST_CONCAT(test_reg_, __LINE__)(name, EMBER_TEST_CONCAT(test_fn_, __LINE__)); \
    static void EMBER_TEST_CONCAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                                                            \
    do {                                                                                                       \
        if (!(cond)) throw ::ember::test::Failure{std::format("{}:{}: CHECK({}) failed", __FILE__, __LINE__, #cond)}; \
    } while (0)

#define CHECK_MSG(cond, ...)                                                                              \
    do {                                                                                                  \
        if (!(cond))                                                                                      \
            throw ::ember::test::Failure{std::format("{}:{}: {}", __FILE__, __LINE__, std::format(__VA_ARGS__))}; \
    } while (0)

#define CHECK_EQ(a, b)                                                                                     \
    do {                                                                                                   \
        const auto va_ = (a); /* a copy: the argument may refer into a temporary */                       \
        const auto vb_ = (b);                                                                              \
        if (!(va_ == vb_))                                                                                 \
            throw ::ember::test::Failure{std::format("{}:{}: {} == {} failed", __FILE__, __LINE__, #a, #b)}; \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                       \
    do {                                                                                                            \
        double va_ = (a), vb_ = (b);                                                                                \
        if (!(std::fabs(va_ - vb_) <= (tol)))                                                                       \
            throw ::ember::test::Failure{std::format("{}:{}: {} = {} is not within {} of {}", __FILE__, __LINE__, #a, va_, (tol), vb_)}; \
    } while (0)

#define CHECK_THROWS(expr)                                                                                    \
    do {                                                                                                      \
        bool threw_ = false;                                                                                  \
        try {                                                                                                 \
            (void)(expr);                                                                                     \
        } catch (const ::ember::Error &) {                                                                    \
            threw_ = true;                                                                                    \
        }                                                                                                     \
        if (!threw_) throw ::ember::test::Failure{std::format("{}:{}: {} did not throw", __FILE__, __LINE__, #expr)}; \
    } while (0)
