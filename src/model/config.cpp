#include "model/config.hpp"

#include <format>

#include "common/error.hpp"
#include "common/mmap_file.hpp"

namespace ember {

int64_t ModelConfig::parameters() const {
    int64_t attn = static_cast<int64_t>(hidden) * (q_dim() + 2 * kv_dim()) + static_cast<int64_t>(q_dim()) * hidden;
    int64_t mlp = 3LL * hidden * intermediate;
    int64_t norms = 2LL * hidden + (qk_norm ? 2LL * head_dim : 0);
    int64_t embed = static_cast<int64_t>(vocab_size) * hidden;
    return n_layers * (attn + mlp + norms) + embed * (tie_embeddings ? 1 : 2) + hidden;
}

ModelConfig ModelConfig::from_json(const json::Value &c, const json::Value *gen) {
    ModelConfig m;
    m.arch = c.get_string("model_type", "");
    if (m.arch != "qwen3" && m.arch != "qwen2")
        fail("unsupported model_type \"{}\" (supported: qwen3, qwen2)", m.arch);
    m.vocab_size = static_cast<int>(c["vocab_size"].as_int());
    m.hidden = static_cast<int>(c["hidden_size"].as_int());
    m.intermediate = static_cast<int>(c["intermediate_size"].as_int());
    m.n_layers = static_cast<int>(c["num_hidden_layers"].as_int());
    m.n_heads = static_cast<int>(c["num_attention_heads"].as_int());
    m.n_kv_heads = static_cast<int>(c.get_int("num_key_value_heads", m.n_heads));
    m.head_dim = static_cast<int>(c.get_int("head_dim", m.hidden / m.n_heads));
    m.rms_eps = static_cast<float>(c.get_number("rms_norm_eps", 1e-6));
    m.rope_theta = c.get_number("rope_theta", 10000.0);
    m.max_position = static_cast<int>(c.get_int("max_position_embeddings", 4096));
    m.tie_embeddings = c.get_bool("tie_word_embeddings", false);
    m.qk_norm = m.arch == "qwen3";
    m.qkv_bias = m.arch == "qwen2" || c.get_bool("attention_bias", false);

    if (const json::Value *rs = c.find("rope_scaling"); rs && !rs->is_null())
        fail("rope_scaling is not supported yet");
    if (m.hidden <= 0 || m.n_layers <= 0 || m.n_heads <= 0 || m.n_kv_heads <= 0 || m.head_dim <= 0)
        fail("config.json has non-positive dimensions");
    if (m.n_heads % m.n_kv_heads != 0) fail("num_attention_heads must be a multiple of num_key_value_heads");
    if (m.head_dim % 2 != 0 || m.head_dim > 256) fail("head_dim must be even and at most 256");

    auto add_eos = [&](const json::Value *v) {
        if (!v || v->is_null()) return;
        if (v->is_array())
            for (const auto &e : v->as_array()) m.eos_ids.push_back(static_cast<int>(e.as_int()));
        else
            m.eos_ids.push_back(static_cast<int>(v->as_int()));
    };
    add_eos(gen ? gen->find("eos_token_id") : nullptr);
    if (m.eos_ids.empty()) add_eos(c.find("eos_token_id"));
    return m;
}

ModelConfig ModelConfig::load(const std::string &dir) {
    json::Value c = json::Value::parse(read_file(dir + "/config.json"));
    json::Value gen;
    std::string gen_path = dir + "/generation_config.json";
    if (file_exists(gen_path)) gen = json::Value::parse(read_file(gen_path));
    return from_json(c, gen.is_null() ? nullptr : &gen);
}

std::string ModelConfig::describe() const {
    return std::format("{} with {:.2f}B parameters: {} layers, d={}, {} heads ({} kv) x {}, mlp {}, vocab {}",
                       arch, static_cast<double>(parameters()) / 1e9, n_layers, hidden, n_heads, n_kv_heads,
                       head_dim, intermediate, vocab_size);
}

}  // namespace ember
