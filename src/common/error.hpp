// Errors. Setup and I/O failures (a missing model file, a malformed config,
// out of GPU memory at startup) throw ember::Error; the per-step hot path
// never throws.
#pragma once

#include <format>
#include <stdexcept>
#include <string>

namespace ember {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &msg) : std::runtime_error(msg) {}
};

template <class... Args>
[[noreturn]] inline void fail(std::format_string<Args...> fmt, Args &&...args) {
    throw Error(std::format(fmt, std::forward<Args>(args)...));
}

#define EMBER_CHECK(cond, ...)                 \
    do {                                       \
        if (!(cond)) ::ember::fail(__VA_ARGS__); \
    } while (0)

}  // namespace ember
