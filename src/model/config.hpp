// The model's hyperparameters, read from the Hugging Face config.json.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/json.hpp"

namespace ember {

struct ModelConfig {
    std::string arch;          // "qwen3", "qwen2", "llama"
    int vocab_size = 0;        // rows of the embedding matrix (may exceed the tokenizer's vocabulary)
    int hidden = 0;            // d_model
    int intermediate = 0;      // MLP inner size
    int n_layers = 0;
    int n_heads = 0;           // query heads
    int n_kv_heads = 0;        // key/value heads (GQA)
    int head_dim = 0;
    float rms_eps = 1e-6f;
    double rope_theta = 10000.0;
    int max_position = 0;
    bool tie_embeddings = false;
    bool qk_norm = false;      // Qwen3: RMSNorm on each query/key head before RoPE
    bool qkv_bias = false;     // Qwen2: biases on the q/k/v projections
    std::vector<int> eos_ids;

    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int group() const { return n_heads / n_kv_heads; }  // query heads per kv head

    // Parameter count, for logs and memory planning.
    int64_t parameters() const;
    // Bytes of K+V cache per token, all layers, at `elem_bytes` per element.
    int64_t kv_bytes_per_token(int elem_bytes) const {
        return 2LL * n_layers * kv_dim() * elem_bytes;
    }

    static ModelConfig from_json(const json::Value &config, const json::Value *generation_config);
    static ModelConfig load(const std::string &model_dir);  // reads config.json (+ generation_config.json)
    std::string describe() const;
};

}  // namespace ember
