// The interface between the engine and the hardware.
//
// A backend owns the model weights and the paged KV cache, and runs one
// forward step over a "ragged" batch: S sequences, each contributing a
// contiguous run of new tokens. A run can be a chunk of a prompt (prefill), the
// single newest token (decode), or the k+1 tokens of a speculative check - all
// three are the same operation, which is what lets the scheduler mix them
// freely in one step.
//
// The KV cache is split into blocks of kBlockSize tokens. Each sequence has a
// block table mapping its logical blocks to physical ones; the engine owns the
// tables, the backend only reads them.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "model/config.hpp"

namespace ember {

constexpr int kBlockSize = 16;

enum class WeightFormat { f16, int8, int4 };
const char *weight_format_name(WeightFormat f);
WeightFormat parse_weight_format(const std::string &s);  // "f16" | "int8" | "int4"

struct SamplingParams {
    float temperature = 1.0f;  // 0 = greedy
    float top_p = 1.0f;
    int top_k = 0;             // 0 = no limit
    float min_p = 0.0f;        // drop tokens below min_p * p(best)
    uint64_t seed = 0;
};

struct BackendOptions {
    std::string device = "cuda";    // "cuda" or "cpu"
    WeightFormat weights = WeightFormat::f16;
    // Per-matrix overrides of `weights`, e.g. "down=int8,qkv=f16" (kinds: qkv, o, gate_up, down, lm_head).
    std::string quant_mix;
    int threads = 0;                // CPU backend: 0 = all cores
    int max_batch_tokens = 2048;    // tokens per forward step (sizes activation buffers)
    int max_seqs = 64;              // sequences per step
    int max_logit_rows = 0;         // 0 = max_seqs; more for speculative verification
    double memory_fraction = 0.92;  // CUDA: share of free memory to use (weights + KV + buffers)
    int64_t kv_cache_bytes = 0;     // explicit KV cache size; 0 = whatever memory_fraction leaves
    int kv_cache_tokens = 0;        // CPU backend: KV capacity in tokens; 0 = 8192
    bool cuda_graphs = true;
    bool cublas = false;            // CUDA: use cuBLAS for large GEMMs (for comparison)
    int gpu = 0;
};

// One forward step. Arrays indexed by token have T entries, by sequence S.
struct StepBatch {
    std::vector<int32_t> tokens;        // [T] token ids
    std::vector<int32_t> positions;     // [T] absolute position of each token in its sequence
    std::vector<int32_t> query_start;   // [S+1] sequence s owns tokens [query_start[s], query_start[s+1])
    std::vector<int32_t> context_len;   // [S] tokens in the KV cache after this step (last position + 1)
    std::vector<int32_t> block_tables;  // [S * max_blocks] physical block of each logical block
    int max_blocks = 0;
    std::vector<int32_t> logit_rows;    // token indices whose logits are wanted, in output order

    int num_tokens() const { return static_cast<int>(tokens.size()); }
    int num_seqs() const { return static_cast<int>(context_len.size()); }
    void clear() {
        tokens.clear();
        positions.clear();
        query_start.assign(1, 0);
        context_len.clear();
        block_tables.clear();
        max_blocks = 0;
        logit_rows.clear();
    }
};

// Per logit row: how to pick the next token. `counter` makes the random draw a
// pure function of (seed, counter), so a sequence samples the same tokens
// whatever else happens to share its batch.
struct SampleRequest {
    SamplingParams params;
    uint64_t counter = 0;
};

struct MemoryInfo {
    int64_t weights = 0;
    int64_t kv_cache = 0;
    int64_t activations = 0;
    int64_t device_total = 0;
    int64_t device_free = 0;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string name() const = 0;
    virtual const ModelConfig &config() const = 0;
    virtual const BackendOptions &options() const = 0;
    virtual int num_kv_blocks() const = 0;
    virtual MemoryInfo memory() const = 0;

    // Runs the model on the batch: appends the new tokens' keys and values to
    // the cache and computes logits for batch.logit_rows (kept on the device).
    virtual void forward(const StepBatch &batch) = 0;

    // One token per logit row of the last forward(), in order.
    virtual void sample(std::span<const SampleRequest> requests, std::span<int32_t> out) = 0;

    // Probability distributions for the logit rows (after temperature/top-k/top-p),
    // for speculative decoding: probs(row, token) for the listed (row, token) pairs.
    virtual void token_probs(std::span<const SampleRequest> requests, std::span<const int32_t> rows,
                             std::span<const int32_t> tokens, std::span<float> out) = 0;

    // Copies logits of one row of the last forward() to the host.
    virtual void logits(int row, std::span<float> out) = 0;

    // Debugging: when enabled, forward() records the residual stream of the
    // last token after the embedding and after every layer.
    virtual void set_capture_hidden(bool on) = 0;
    virtual std::vector<float> captured_hidden() = 0;  // [(n_layers + 1) * hidden]
};

std::unique_ptr<Backend> create_backend(const std::string &model_dir, const BackendOptions &options);

bool cuda_available();

}  // namespace ember
