#include "backend/backend.hpp"

#include "common/error.hpp"

namespace ember {

std::unique_ptr<Backend> create_cpu_backend(const std::string &dir, const BackendOptions &opt);
#ifdef EMBER_WITH_CUDA
std::unique_ptr<Backend> create_cuda_backend(const std::string &dir, const BackendOptions &opt);
bool cuda_device_present();
#endif

const char *weight_format_name(WeightFormat f) {
    switch (f) {
        case WeightFormat::f16: return "f16";
        case WeightFormat::int8: return "int8";
        case WeightFormat::int4: return "int4";
    }
    return "?";
}

WeightFormat parse_weight_format(const std::string &s) {
    if (s == "f16" || s == "fp16") return WeightFormat::f16;
    if (s == "int8" || s == "q8") return WeightFormat::int8;
    if (s == "int4" || s == "q4") return WeightFormat::int4;
    fail("unknown weight format \"{}\" (use f16, int8 or int4)", s);
}

int64_t estimate_weight_bytes(const std::string &dir, const BackendOptions &opt) {
    ModelConfig c = ModelConfig::load(dir);
    const int64_t H = c.hidden, I = c.intermediate, L = c.n_layers;
    const int64_t qkv = (c.q_dim() + 2 * c.kv_dim()) * H, o = H * c.q_dim(), mlp_up = 2 * I * H, down = H * I;
    const int64_t head = static_cast<int64_t>(c.vocab_size) * H;
    switch (opt.weights) {
        case WeightFormat::f16: return 2 * (L * (qkv + o + mlp_up + down) + head * (c.tie_embeddings ? 1 : 2));
        case WeightFormat::int8: return L * (qkv + o + mlp_up + down) * 103 / 100 + head;
        case WeightFormat::int4: return L * ((qkv + down) * 103 / 100 + (o + mlp_up) * 53 / 100) + head;
    }
    return 0;
}

#ifdef EMBER_WITH_CUDA
int64_t cuda_free_bytes();
#endif

int64_t gpu_free_memory() {
#ifdef EMBER_WITH_CUDA
    return cuda_free_bytes();
#else
    return 0;
#endif
}

int64_t plan_kv_split(const std::string &target_dir, const std::string &draft_dir, const BackendOptions &opt,
                      const BackendOptions &draft) {
    const int64_t free = gpu_free_memory();
    const int64_t weights = estimate_weight_bytes(target_dir, opt) + estimate_weight_bytes(draft_dir, draft);
    // Activation buffers and scratch of both, plus the target's logits for k + 1 rows per sequence.
    const int64_t logit_rows = opt.max_logit_rows > 0 ? opt.max_logit_rows : opt.max_seqs;
    const int64_t buffers = (300ll << 20) + (200ll << 20) + logit_rows * ModelConfig::load(target_dir).vocab_size * 4;
    const int64_t keep = static_cast<int64_t>((1.0 - opt.memory_fraction) * static_cast<double>(free)) + (128ll << 20);
    const int64_t left = free - weights - buffers - keep;
    if (left < (2ll * 64 << 20))
        fail("not enough GPU memory for both models ({:.0f} MiB free, {:.0f} MiB of weights); try -q int4 or --draft-weights int4",
             static_cast<double>(free) / (1 << 20), static_cast<double>(weights) / (1 << 20));
    return left / 2;
}

bool cuda_available() {
#ifdef EMBER_WITH_CUDA
    return cuda_device_present();
#else
    return false;
#endif
}

std::unique_ptr<Backend> create_backend(const std::string &dir, const BackendOptions &opt) {
    if (opt.device == "cpu") return create_cpu_backend(dir, opt);
    if (opt.device == "cuda") {
#ifdef EMBER_WITH_CUDA
        return create_cuda_backend(dir, opt);
#else
        fail("this build of Ember has no CUDA support; use --device cpu");
#endif
    }
    fail("unknown device \"{}\" (use cuda or cpu)", opt.device);
}

}  // namespace ember
