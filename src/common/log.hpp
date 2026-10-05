// Minimal thread-safe logging to stderr.
#pragma once

#include <format>
#include <string>

namespace ember::log {

enum class Level { debug = 0, info = 1, warn = 2, error = 3 };

void set_level(Level level);
Level level();
void write(Level level, const std::string &msg);

template <class... Args>
void debug(std::format_string<Args...> fmt, Args &&...args) {
    if (level() <= Level::debug) write(Level::debug, std::format(fmt, std::forward<Args>(args)...));
}
template <class... Args>
void info(std::format_string<Args...> fmt, Args &&...args) {
    if (level() <= Level::info) write(Level::info, std::format(fmt, std::forward<Args>(args)...));
}
template <class... Args>
void warn(std::format_string<Args...> fmt, Args &&...args) {
    if (level() <= Level::warn) write(Level::warn, std::format(fmt, std::forward<Args>(args)...));
}
template <class... Args>
void error(std::format_string<Args...> fmt, Args &&...args) {
    write(Level::error, std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace ember::log
