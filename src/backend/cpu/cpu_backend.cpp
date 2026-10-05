// The CPU backend: a direct, readable implementation of the transformer in
// f32, used as the reference the CUDA kernels are checked against, and to run
// the engine without a GPU (tests, CI).
#include <cmath>
#include <cstring>
#include <format>

#include "backend/backend.hpp"
#include "backend/cpu/cpu_kernels.hpp"
#include "backend/sampling_host.hpp"
#include "common/error.hpp"
#include "common/log.hpp"
#include "common/thread_pool.hpp"
#include "model/safetensors.hpp"

namespace ember {

namespace {

using cpu::Weight;

struct Layer {
    std::vector<float> in_norm, post_norm, q_norm, k_norm, q_bias, k_bias, v_bias;
    Weight q, k, v, o, gate, up, down;
};

std::vector<float> load_vector(const SafeTensors &st, const std::string &name, int64_t n) {
    const TensorView &t = st.get(name, {n});
    std::vector<float> v(static_cast<size_t>(n));
    cpu::load_row(Weight::from(t), 0, v.data());
    return v;
}

void rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += static_cast<double>(x[i]) * x[i];
    float scale = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * scale);
}

class CpuBackend final : public Backend {
public:
    CpuBackend(const std::string &dir, const BackendOptions &opt)
        : cfg_(ModelConfig::load(dir)), opt_(opt), st_(SafeTensors::open_dir(dir)), pool_(opt.threads) {
        if (opt_.weights != WeightFormat::f16)
            log::warn("the CPU backend computes in f32 from the file's weights; --weights {} is ignored",
                      weight_format_name(opt_.weights));
        const ModelConfig &c = cfg_;
        const int64_t H = c.hidden, Q = c.q_dim(), KV = c.kv_dim(), I = c.intermediate, D = c.head_dim;
        embed_ = Weight::from(st_.get("model.embed_tokens.weight", {c.vocab_size, H}));
        lm_head_ = c.tie_embeddings || !st_.has("lm_head.weight") ? embed_
                                                                   : Weight::from(st_.get("lm_head.weight", {c.vocab_size, H}));
        final_norm_ = load_vector(st_, "model.norm.weight", H);
        layers_.resize(static_cast<size_t>(c.n_layers));
        for (int l = 0; l < c.n_layers; l++) {
            std::string p = std::format("model.layers.{}.", l);
            Layer &L = layers_[static_cast<size_t>(l)];
            L.in_norm = load_vector(st_, p + "input_layernorm.weight", H);
            L.post_norm = load_vector(st_, p + "post_attention_layernorm.weight", H);
            L.q = Weight::from(st_.get(p + "self_attn.q_proj.weight", {Q, H}));
            L.k = Weight::from(st_.get(p + "self_attn.k_proj.weight", {KV, H}));
            L.v = Weight::from(st_.get(p + "self_attn.v_proj.weight", {KV, H}));
            L.o = Weight::from(st_.get(p + "self_attn.o_proj.weight", {H, Q}));
            if (c.qk_norm) {
                L.q_norm = load_vector(st_, p + "self_attn.q_norm.weight", D);
                L.k_norm = load_vector(st_, p + "self_attn.k_norm.weight", D);
            }
            if (c.qkv_bias) {
                L.q_bias = load_vector(st_, p + "self_attn.q_proj.bias", Q);
                L.k_bias = load_vector(st_, p + "self_attn.k_proj.bias", KV);
                L.v_bias = load_vector(st_, p + "self_attn.v_proj.bias", KV);
            }
            L.gate = Weight::from(st_.get(p + "mlp.gate_proj.weight", {I, H}));
            L.up = Weight::from(st_.get(p + "mlp.up_proj.weight", {I, H}));
            L.down = Weight::from(st_.get(p + "mlp.down_proj.weight", {H, I}));
        }

        int tokens = opt_.kv_cache_tokens > 0 ? opt_.kv_cache_tokens : 4096;
        num_blocks_ = (tokens + kBlockSize - 1) / kBlockSize;
        block_elems_ = static_cast<size_t>(c.n_kv_heads) * kBlockSize * D;
        size_t cache = static_cast<size_t>(c.n_layers) * num_blocks_ * block_elems_;
        k_cache_.assign(cache, 0.0f);
        v_cache_.assign(cache, 0.0f);

        inv_freq_.resize(static_cast<size_t>(D / 2));
        for (int i = 0; i < D / 2; i++)  // as transformers computes it: in float32
            inv_freq_[static_cast<size_t>(i)] =
                1.0f / std::pow(static_cast<float>(c.rope_theta), static_cast<float>(2 * i) / static_cast<float>(D));

        log::info("cpu backend: {} | {} threads{} | KV cache {} tokens ({:.0f} MiB)", c.describe(), pool_.size(),
                  cpu::has_avx2() ? ", AVX2" : "", num_blocks_ * kBlockSize,
                  static_cast<double>(2 * cache * sizeof(float)) / (1 << 20));
    }

    std::string name() const override { return "cpu"; }
    const ModelConfig &config() const override { return cfg_; }
    const BackendOptions &options() const override { return opt_; }
    int num_kv_blocks() const override { return num_blocks_; }

    MemoryInfo memory() const override {
        MemoryInfo m;
        m.weights = static_cast<int64_t>(st_.total_bytes());
        m.kv_cache = static_cast<int64_t>(2 * k_cache_.size() * sizeof(float));
        return m;
    }

    void forward(const StepBatch &b) override {
        const ModelConfig &c = cfg_;
        const int T = b.num_tokens(), S = b.num_seqs();
        const int H = c.hidden, Q = c.q_dim(), KV = c.kv_dim(), I = c.intermediate, D = c.head_dim;
        EMBER_CHECK(T > 0 && static_cast<int>(b.query_start.size()) == S + 1, "malformed batch");
        EMBER_CHECK(T <= opt_.max_batch_tokens, "batch has {} tokens; the limit is {}", T, opt_.max_batch_tokens);

        seq_of_.resize(static_cast<size_t>(T));
        for (int s = 0; s < S; s++)
            for (int t = b.query_start[s]; t < b.query_start[s + 1]; t++) seq_of_[static_cast<size_t>(t)] = s;

        auto grow = [](std::vector<float> &v, size_t n) {
            if (v.size() < n) v.resize(n);
        };
        grow(h_, static_cast<size_t>(T) * H);
        grow(x_, static_cast<size_t>(T) * H);
        grow(q_, static_cast<size_t>(T) * Q);
        grow(k_, static_cast<size_t>(T) * KV);
        grow(v_, static_cast<size_t>(T) * KV);
        grow(att_, static_cast<size_t>(T) * Q);
        grow(gate_, static_cast<size_t>(T) * I);
        grow(up_, static_cast<size_t>(T) * I);
        if (capture_) captured_.assign(static_cast<size_t>(c.n_layers + 1) * H, 0.0f);

        for (int t = 0; t < T; t++) {
            int tok = b.tokens[static_cast<size_t>(t)];
            EMBER_CHECK(tok >= 0 && tok < c.vocab_size, "token id {} out of range", tok);
            cpu::load_row(embed_, tok, h_.data() + static_cast<size_t>(t) * H);
        }
        capture(0, T, H);

        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (int l = 0; l < c.n_layers; l++) {
            const Layer &L = layers_[static_cast<size_t>(l)];
            for (int t = 0; t < T; t++)
                rmsnorm(&h_[static_cast<size_t>(t) * H], L.in_norm.data(), H, c.rms_eps, &x_[static_cast<size_t>(t) * H]);
            cpu::matmul(pool_, x_.data(), T, L.q, q_.data());
            cpu::matmul(pool_, x_.data(), T, L.k, k_.data());
            cpu::matmul(pool_, x_.data(), T, L.v, v_.data());
            if (c.qkv_bias) {
                for (int t = 0; t < T; t++) {
                    for (int i = 0; i < Q; i++) q_[static_cast<size_t>(t) * Q + i] += L.q_bias[static_cast<size_t>(i)];
                    for (int i = 0; i < KV; i++) {
                        k_[static_cast<size_t>(t) * KV + i] += L.k_bias[static_cast<size_t>(i)];
                        v_[static_cast<size_t>(t) * KV + i] += L.v_bias[static_cast<size_t>(i)];
                    }
                }
            }
            // QK norm, RoPE, then store keys and values in the paged cache.
            for (int t = 0; t < T; t++) {
                int pos = b.positions[static_cast<size_t>(t)];
                for (int h = 0; h < c.n_heads; h++) {
                    float *q = &q_[static_cast<size_t>(t) * Q + static_cast<size_t>(h) * D];
                    if (c.qk_norm) rmsnorm(q, L.q_norm.data(), D, c.rms_eps, q);
                    rope(q, pos);
                }
                for (int h = 0; h < c.n_kv_heads; h++) {
                    float *k = &k_[static_cast<size_t>(t) * KV + static_cast<size_t>(h) * D];
                    if (c.qk_norm) rmsnorm(k, L.k_norm.data(), D, c.rms_eps, k);
                    rope(k, pos);
                    const float *v = &v_[static_cast<size_t>(t) * KV + static_cast<size_t>(h) * D];
                    size_t off = cache_offset(b, seq_of_[static_cast<size_t>(t)], l, h, pos);
                    std::memcpy(&k_cache_[off], k, sizeof(float) * D);
                    std::memcpy(&v_cache_[off], v, sizeof(float) * D);
                }
            }
            // Causal attention over the cache: token t sees positions 0..pos(t).
            pool_.parallel_for(static_cast<size_t>(T) * c.n_heads, 4, [&](size_t i0, size_t i1) {
                std::vector<float> scores;
                for (size_t i = i0; i < i1; i++) {
                    int t = static_cast<int>(i / c.n_heads), h = static_cast<int>(i % c.n_heads);
                    int s = seq_of_[static_cast<size_t>(t)], kvh = h / c.group();
                    int ctx = b.positions[static_cast<size_t>(t)] + 1;
                    const float *q = &q_[static_cast<size_t>(t) * Q + static_cast<size_t>(h) * D];
                    scores.resize(static_cast<size_t>(ctx));
                    float mx = -INFINITY;
                    for (int j = 0; j < ctx; j++) {
                        float sc = cpu::dot(q, &k_cache_[cache_offset(b, s, l, kvh, j)], D) * scale;
                        scores[static_cast<size_t>(j)] = sc;
                        mx = std::max(mx, sc);
                    }
                    double sum = 0;
                    for (int j = 0; j < ctx; j++) sum += scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - mx);
                    float *out = &att_[static_cast<size_t>(t) * Q + static_cast<size_t>(h) * D];
                    std::fill_n(out, D, 0.0f);
                    for (int j = 0; j < ctx; j++) {
                        float w = static_cast<float>(scores[static_cast<size_t>(j)] / sum);
                        const float *v = &v_cache_[cache_offset(b, s, l, kvh, j)];
                        for (int d = 0; d < D; d++) out[d] += w * v[d];
                    }
                }
            });
            cpu::matmul(pool_, att_.data(), T, L.o, x_.data());
            for (size_t i = 0; i < static_cast<size_t>(T) * H; i++) h_[i] += x_[i];

            for (int t = 0; t < T; t++)
                rmsnorm(&h_[static_cast<size_t>(t) * H], L.post_norm.data(), H, c.rms_eps, &x_[static_cast<size_t>(t) * H]);
            cpu::matmul(pool_, x_.data(), T, L.gate, gate_.data());
            cpu::matmul(pool_, x_.data(), T, L.up, up_.data());
            for (size_t i = 0; i < static_cast<size_t>(T) * I; i++) {
                float g = gate_[i];
                gate_[i] = g / (1.0f + std::exp(-g)) * up_[i];
            }
            cpu::matmul(pool_, gate_.data(), T, L.down, x_.data());
            for (size_t i = 0; i < static_cast<size_t>(T) * H; i++) h_[i] += x_[i];
            capture(l + 1, T, H);
        }

        const int R = static_cast<int>(b.logit_rows.size());
        rows_ = R;
        std::vector<float> normed(static_cast<size_t>(R) * H);
        for (int r = 0; r < R; r++) {
            int t = b.logit_rows[static_cast<size_t>(r)];
            EMBER_CHECK(t >= 0 && t < T, "logit row {} out of range", t);
            rmsnorm(&h_[static_cast<size_t>(t) * H], final_norm_.data(), H, c.rms_eps, &normed[static_cast<size_t>(r) * H]);
        }
        logits_.resize(static_cast<size_t>(R) * c.vocab_size);
        if (R > 0) cpu::matmul(pool_, normed.data(), R, lm_head_, logits_.data());
    }

    void sample(std::span<const SampleRequest> reqs, std::span<int32_t> out) override {
        EMBER_CHECK(static_cast<int>(reqs.size()) == rows_ && out.size() == reqs.size(), "sample: {} requests for {} rows",
                    reqs.size(), rows_);
        for (size_t r = 0; r < reqs.size(); r++)
            out[r] = sample_host(&logits_[r * static_cast<size_t>(cfg_.vocab_size)], cfg_.vocab_size, reqs[r]);
    }

    void token_probs(std::span<const SampleRequest> reqs, std::span<const int32_t> rows, std::span<const int32_t> tokens,
                     std::span<float> out) override {
        for (size_t i = 0; i < rows.size(); i++) {
            int r = rows[i];
            auto dist = sampling_distribution(&logits_[static_cast<size_t>(r) * cfg_.vocab_size], cfg_.vocab_size,
                                              reqs[static_cast<size_t>(r)].params);
            float p = 0;
            for (const auto &[tok, prob] : dist)
                if (tok == tokens[i]) p = prob;
            out[i] = p;
        }
    }

    void logits(int row, std::span<float> out) override {
        EMBER_CHECK(row >= 0 && row < rows_ && static_cast<int>(out.size()) >= cfg_.vocab_size, "logits: bad row {}", row);
        std::memcpy(out.data(), &logits_[static_cast<size_t>(row) * cfg_.vocab_size], sizeof(float) * cfg_.vocab_size);
    }

    void set_capture_hidden(bool on) override { capture_ = on; }
    std::vector<float> captured_hidden() override { return captured_; }

private:
    size_t cache_offset(const StepBatch &b, int seq, int layer, int kv_head, int pos) const {
        int block = b.block_tables[static_cast<size_t>(seq) * b.max_blocks + pos / kBlockSize];
        return ((static_cast<size_t>(layer) * num_blocks_ + block) * cfg_.n_kv_heads + kv_head) * kBlockSize * cfg_.head_dim +
               static_cast<size_t>(pos % kBlockSize) * cfg_.head_dim;
    }

    // Rotary embedding, "rotate half" layout: pairs (x[i], x[i + D/2]).
    void rope(float *x, int pos) const {
        const int half = cfg_.head_dim / 2;
        for (int i = 0; i < half; i++) {
            float angle = static_cast<float>(pos) * inv_freq_[static_cast<size_t>(i)];
            float cs = std::cos(angle), sn = std::sin(angle);
            float a = x[i], b = x[i + half];
            x[i] = a * cs - b * sn;
            x[i + half] = b * cs + a * sn;
        }
    }

    void capture(int index, int T, int H) {
        if (!capture_) return;
        std::memcpy(&captured_[static_cast<size_t>(index) * H], &h_[static_cast<size_t>(T - 1) * H], sizeof(float) * H);
    }

    ModelConfig cfg_;
    BackendOptions opt_;
    SafeTensors st_;
    ThreadPool pool_;
    Weight embed_, lm_head_;
    std::vector<float> final_norm_;
    std::vector<Layer> layers_;
    std::vector<float> inv_freq_;

    int num_blocks_ = 0;
    size_t block_elems_ = 0;
    std::vector<float> k_cache_, v_cache_;  // [layer][block][kv_head][slot][head_dim]

    std::vector<int> seq_of_;
    std::vector<float> h_, x_, q_, k_, v_, att_, gate_, up_, logits_;
    int rows_ = 0;
    bool capture_ = false;
    std::vector<float> captured_;
};

}  // namespace

std::unique_ptr<Backend> create_cpu_backend(const std::string &dir, const BackendOptions &opt) {
    return std::make_unique<CpuBackend>(dir, opt);
}

}  // namespace ember
