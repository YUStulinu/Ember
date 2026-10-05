#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "backend/backend.hpp"
#include "common/error.hpp"
#include "common/log.hpp"
#include "common/mmap_file.hpp"
#include "test.hpp"

namespace ember::test {

namespace {
struct Entry {
    const char *name;
    std::function<void()> fn;
};
std::vector<Entry> &registry() {
    static std::vector<Entry> r;
    return r;
}
uint64_t g_seed = 20261005;
bool g_verbose = false;
}  // namespace

void add(const char *name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }

std::string source_dir() {
    if (const char *env = std::getenv("EMBER_SOURCE_DIR")) return env;
    return EMBER_SOURCE_DIR;
}

std::string model_dir(const std::string &name) {
    if (const char *env = std::getenv("EMBER_MODELS")) return std::string(env) + "/" + name;
    return source_dir() + "/models/" + name;
}

void require_model(const std::string &name) {
    if (!file_exists(model_dir(name) + "/config.json")) throw Skip{"model " + name + " not downloaded"};
}

bool has_cuda() { return cuda_available(); }

void require_cuda() {
    if (!has_cuda()) throw Skip{"no CUDA device"};
}

uint64_t seed() { return g_seed; }
bool verbose() { return g_verbose; }

}  // namespace ember::test

int main(int argc, char **argv) {
    using namespace ember::test;
    std::vector<const char *> filters;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) g_seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) g_verbose = true;
        else filters.push_back(argv[i]);
    }
    ember::log::set_level(g_verbose ? ember::log::Level::info : ember::log::Level::warn);

    int run = 0, failed = 0, skipped = 0;
    for (const auto &e : registry()) {
        bool match = filters.empty();
        for (const char *f : filters)
            if (std::strstr(e.name, f)) match = true;
        if (!match) continue;
        std::printf("  %-58s ", e.name);
        std::fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        try {
            e.fn();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("ok (%.0f ms)\n", ms);
            run++;
        } catch (const Skip &s) {
            std::printf("skipped: %s\n", s.reason.c_str());
            skipped++;
        } catch (const Failure &f) {
            std::printf("\n    FAILED %s\n", f.message.c_str());
            failed++;
            run++;
        } catch (const std::exception &ex) {
            std::printf("\n    FAILED with exception: %s\n", ex.what());
            failed++;
            run++;
        }
    }
    std::printf("\n%d tests, %d failed, %d skipped (seed %llu)\n", run, failed, skipped,
                static_cast<unsigned long long>(g_seed));
    return failed ? 1 : 0;
}
