// Model correctness: Ember's forward pass against PyTorch (tests/golden),
// layer by layer, then greedy generation token by token.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "backend/backend.hpp"
#include "common/json.hpp"
#include "common/mmap_file.hpp"
#include "model_helpers.hpp"
#include "test.hpp"

using namespace ember;

namespace {

struct Golden {
    json::Value meta;
    std::vector<float> hidden, logits;
};

Golden load_golden(const std::string &model) {
    std::string dir = test::source_dir() + "/tests/golden/" + model;
    if (!file_exists(dir + "/meta.json")) throw test::Skip{"run reference/golden.py to create golden data"};
    Golden g;
    g.meta = json::Value::parse(read_file(dir + "/meta.json"));
    auto load = [](const std::string &path) {
        std::string raw = read_file(path);
        std::vector<float> v(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), v.size() * 4);
        return v;
    };
    g.hidden = load(dir + "/hidden.f32");
    g.logits = load(dir + "/logits.f32");
    return g;
}

std::vector<int> int_list(const json::Value &arr) {
    std::vector<int> out;
    for (const auto &v : arr.as_array()) out.push_back(static_cast<int>(v.as_int()));
    return out;
}

// Relative error of a against b: ||a - b|| / ||b||.
double rel_error(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

void check_against_golden(const std::string &model, const std::string &device, double hidden_tol, double logit_tol,
                          int greedy_tokens) {
    test::require_model(model);
    Golden g = load_golden(model);
    BackendOptions opt;
    opt.device = device;
    opt.kv_cache_tokens = 512;
    auto be = create_backend(test::model_dir(model), opt);
    const ModelConfig &c = be->config();
    std::vector<int> prompt = int_list(g.meta["prompt_ids"]);

    // 1. One prefill step; compare the residual stream after every layer.
    be->set_capture_hidden(true);
    test::SingleSequence seq(*be);
    seq.prefill(prompt);
    std::vector<float> hidden = be->captured_hidden();
    CHECK_EQ(hidden.size(), g.hidden.size());
    double worst = 0;
    int worst_layer = 0;
    for (int l = 0; l <= c.n_layers; l++) {
        double e = rel_error(&hidden[static_cast<size_t>(l) * c.hidden], &g.hidden[static_cast<size_t>(l) * c.hidden],
                             static_cast<size_t>(c.hidden));
        if (test::verbose() && (l < 4 || l % 7 == 0 || l == c.n_layers)) std::printf("\n      layer %2d: %.2e", l, e);
        if (e > worst) {
            worst = e;
            worst_layer = l;
        }
    }
    if (test::verbose()) std::printf("[worst hidden error %.2e at layer %d] ", worst, worst_layer);
    CHECK_MSG(worst < hidden_tol, "hidden state after layer {} differs by {:.3e}", worst_layer, worst);

    // 2. Logits of the last prompt token: close, and the same top choices.
    std::vector<float> logits(static_cast<size_t>(c.vocab_size));
    be->logits(0, logits);
    double le = rel_error(logits.data(), g.logits.data(), logits.size());
    if (test::verbose()) std::printf("[logit error %.2e] ", le);
    CHECK_MSG(le < logit_tol, "logits differ by {:.3e}", le);
    std::vector<int> top = int_list(g.meta["top_ids"]);
    int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    CHECK_EQ(best, top[0]);

    // 3. Greedy generation, token by token through the paged cache.
    be->set_capture_hidden(false);
    std::vector<int> want = int_list(g.meta["greedy_ids"]);
    std::vector<int> got = seq.greedy(static_cast<int>(std::min<size_t>(want.size(), static_cast<size_t>(greedy_tokens))));
    for (size_t i = 0; i < got.size(); i++)
        CHECK_MSG(got[i] == want[i], "greedy token {} is {}, expected {}", i, got[i], want[i]);
}

}  // namespace

TEST("model cpu: Qwen3-0.6B matches PyTorch layer by layer, then greedily") {
    check_against_golden("Qwen3-0.6B", "cpu", 1e-4, 1e-4, 48);
}

TEST("model cpu: Qwen3-1.7B matches PyTorch") {
    check_against_golden("Qwen3-1.7B", "cpu", 1e-4, 1e-4, 8);
}

TEST("model cuda: Qwen3-0.6B matches PyTorch (f16 tolerances)") {
    test::require_cuda();
    check_against_golden("Qwen3-0.6B", "cuda", 2e-2, 2e-2, 16);
}

TEST("model cuda: Qwen3-1.7B matches PyTorch (f16 tolerances)") {
    test::require_cuda();
    check_against_golden("Qwen3-1.7B", "cuda", 2e-2, 2e-2, 16);
}

TEST("model cuda: long prompts in uneven chunks match the CPU backend") {
    test::require_cuda();
    test::require_model("Qwen3-0.6B");
    BackendOptions copt, gopt;
    copt.device = "cpu";
    copt.kv_cache_tokens = 1024;
    gopt.device = "cuda";
    gopt.kv_cache_bytes = 256ll << 20;
    auto cpu = create_backend(test::model_dir("Qwen3-0.6B"), copt);
    auto gpu = create_backend(test::model_dir("Qwen3-0.6B"), gopt);
    std::vector<int> prompt;
    uint32_t x = 12345;
    for (int i = 0; i < 700; i++) {
        x = x * 1664525u + 1013904223u;
        prompt.push_back(static_cast<int>(1000 + (x >> 8) % 50000));
    }
    const int V = cpu->config().vocab_size;
    std::vector<float> lc(static_cast<size_t>(V)), lg(static_cast<size_t>(V));
    // Chunks of 300, 37 (crosses into the middle of a page) and 363: tiles of 64 and partial tiles.
    test::SingleSequence sc(*cpu), sg(*gpu, 3);  // different physical blocks on purpose
    for (int chunk : {300, 37, 363}) {
        std::vector<int> part(prompt.begin() + sc.length(), prompt.begin() + sc.length() + chunk);
        sc.step(part);
        sg.step(part);
        cpu->logits(0, lc);
        gpu->logits(0, lg);
        double e = rel_error(lg.data(), lc.data(), lc.size());
        if (test::verbose()) std::printf("[after %d tokens: %.2e] ", sc.length(), e);
        CHECK_MSG(e < 3e-2, "logits after {} tokens differ by {:.3e}", sc.length(), e);
    }
    // Then decode a few tokens through the paged kernel with split contexts.
    auto a = sc.greedy(8), b = sg.greedy(8);
    CHECK(a == b);
}
