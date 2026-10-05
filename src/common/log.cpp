#include "common/log.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>

namespace ember::log {

namespace {
std::atomic<int> g_level{static_cast<int>(Level::info)};
std::mutex g_mutex;
const auto g_start = std::chrono::steady_clock::now();
}  // namespace

void set_level(Level level) { g_level.store(static_cast<int>(level)); }

Level level() { return static_cast<Level>(g_level.load(std::memory_order_relaxed)); }

void write(Level lvl, const std::string &msg) {
    static const char *names[] = {"debug", "info ", "warn ", "error"};
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
    std::lock_guard<std::mutex> lock(g_mutex);
    std::fprintf(stderr, "[%9.3f] %s %s\n", t, names[static_cast<int>(lvl)], msg.c_str());
    std::fflush(stderr);
}

}  // namespace ember::log
