// The CUDA backend: weights in f16 (or int8/int4) on the GPU, a paged f16 KV
// cache, an f32 residual stream, and the kernels in this directory.
//
// Weight layout choices made at load time, so the forward pass is a short
// sequence of large kernels:
//   - q, k and v projections are concatenated into one [Q + 2KV, H] matrix
//     (one GEMM instead of three);
//   - gate and up projections are interleaved row by row into [2I, H], so the
//     GEMM epilogue sees (gate_j, up_j) side by side and writes silu(g) * u.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <unordered_map>

#include "backend/backend.hpp"
#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/cuda_testing.hpp"
#include "backend/cuda/kernels.cuh"
#include "common/error.hpp"
#include "common/log.hpp"
#include "model/safetensors.hpp"

namespace ember {

namespace cuda {

int sm_count() {
    static const int n = [] {
        int dev = 0, count = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev);
        return count > 0 ? count : 1;
    }();
    return n;
}

namespace {

SrcType src_type(DType t) {
    switch (t) {
        case DType::f32: return SrcType::f32;
        case DType::f16: return SrcType::f16;
        case DType::bf16: return SrcType::bf16;
        default: fail("unsupported weight dtype {}", dtype_name(t));
    }
}

// A linear layer's weights: f16 [N, K], or quantized.
struct Linear {
    int N = 0, K = 0;
    half *w = nullptr;
    void *wq = nullptr;
    half *scales = nullptr;
    QuantType qt = QuantType::none;
};

struct LayerWeights {
    Linear qkv, o, gate_up, down;
    float *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *qkv_bias = nullptr;
};

// Bump allocator over one device allocation.
class Arena {
public:
    void reserve(size_t bytes) { buf_.resize(bytes); }
    template <class T>
    T *take(size_t count) {
        size_t bytes = (count * sizeof(T) + 255) & ~size_t(255);
        EMBER_CHECK(used_ + bytes <= buf_.size(), "internal: arena overflow");
        T *p = reinterpret_cast<T *>(buf_.get() + used_);
        used_ += bytes;
        return p;
    }
    size_t used() const { return used_; }

private:
    DeviceBuffer<uint8_t> buf_;
    size_t used_ = 0;
};

size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

class CudaBackend final : public Backend {
public:
    CudaBackend(const std::string &dir, const BackendOptions &opt) : cfg_(ModelConfig::load(dir)), opt_(opt) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) fail("no CUDA device found");
        EMBER_CHECK(opt_.gpu >= 0 && opt_.gpu < count, "GPU {} does not exist ({} found)", opt_.gpu, count);
        CUDA_CHECK(cudaSetDevice(opt_.gpu));
        CUDA_CHECK(cudaGetDeviceProperties(&props_, opt_.gpu));
        EMBER_CHECK(props_.major * 10 + props_.minor >= 75, "Ember needs a GPU with compute capability 7.5 or newer (found {}.{})",
                    props_.major, props_.minor);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        if (opt_.max_logit_rows <= 0) opt_.max_logit_rows = opt_.max_seqs;
        qtype_ = opt_.weights == WeightFormat::int8 ? QuantType::int8
                 : opt_.weights == WeightFormat::int4 ? QuantType::int4
                                                      : QuantType::none;

        load_weights(dir);
        allocate_activations();
        allocate_kv_cache();
        log::info("cuda backend on {} ({} SMs, {:.1f} GiB): {} | weights {:.0f} MiB, KV cache {} tokens ({:.0f} MiB), "
                  "buffers {:.0f} MiB",
                  props_.name, props_.multiProcessorCount, static_cast<double>(props_.totalGlobalMem) / (1 << 30),
                  cfg_.describe(), mib(weight_bytes_), num_blocks_ * kBlockSize, mib(kv_bytes_), mib(act_bytes_));
    }

    ~CudaBackend() override {
        for (auto &[key, g] : graphs_) cudaGraphExecDestroy(g.exec);
        if (stream_) cudaStreamDestroy(stream_);
        if (host_ints_) cudaFreeHost(host_ints_);
        if (host_params_) cudaFreeHost(host_params_);
        if (host_tokens_) cudaFreeHost(host_tokens_);
    }

    std::string name() const override { return "cuda"; }
    const ModelConfig &config() const override { return cfg_; }
    const BackendOptions &options() const override { return opt_; }
    int num_kv_blocks() const override { return num_blocks_; }

    MemoryInfo memory() const override {
        MemoryInfo m;
        m.weights = static_cast<int64_t>(weight_bytes_);
        m.kv_cache = static_cast<int64_t>(kv_bytes_);
        m.activations = static_cast<int64_t>(act_bytes_);
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        m.device_free = static_cast<int64_t>(free);
        m.device_total = static_cast<int64_t>(total);
        return m;
    }

    void forward(const StepBatch &b) override {
        const ModelConfig &c = cfg_;
        const int T = b.num_tokens(), S = b.num_seqs(), R = static_cast<int>(b.logit_rows.size());
        EMBER_CHECK(T > 0 && T <= opt_.max_batch_tokens, "batch of {} tokens (limit {})", T, opt_.max_batch_tokens);
        EMBER_CHECK(S <= opt_.max_seqs && R <= opt_.max_logit_rows, "batch of {} sequences / {} logit rows exceeds the limits",
                    S, R);
        EMBER_CHECK(static_cast<int>(b.block_tables.size()) == S * b.max_blocks && b.max_blocks <= max_blocks_per_seq_,
                    "malformed block tables");
        rows_ = R;

        // A pure decode step (one token per sequence, all needing logits) replays a
        // CUDA graph: ~300 kernel launches become one.
        bool decode_only = T == S && R == T && T <= kGraphMaxTokens && opt_.cuda_graphs && !capture_;
        for (int i = 0; decode_only && i < R; i++) decode_only = b.logit_rows[static_cast<size_t>(i)] == i;
        if (decode_only) {
            forward_graph(b);
            return;
        }

        // Pack the step's integers into one pinned buffer and upload it with one copy.
        int32_t *hp = host_ints_;
        size_t off = 0;
        auto put = [&](const int32_t *src, size_t n) {
            size_t at = off;
            if (n) std::memcpy(hp + off, src, n * sizeof(int32_t));
            off += (n + 31) & ~size_t(31);
            return dev_ints_.get() + at;
        };
        // Attention plan: sequences with many new tokens use the tensor-core prefill
        // kernel in tiles of 64 queries; the rest (decode) use the paged kernel.
        token_seq_.resize(static_cast<size_t>(T));
        tile_start_.clear();
        tile_end_.clear();
        tile_seq_.clear();
        decode_tokens_.clear();
        int max_ctx = 0;
        const bool flash_ok = c.head_dim == 64 || c.head_dim == 128;
        for (int s = 0; s < S; s++) {
            const int q0 = b.query_start[s], q1 = b.query_start[s + 1];
            for (int t = q0; t < q1; t++) token_seq_[static_cast<size_t>(t)] = s;
            if (flash_ok && q1 - q0 >= kFlashMinTokens) {
                for (int t = q0; t < q1; t += kPrefillTile) {
                    tile_start_.push_back(t);
                    tile_end_.push_back(std::min(q1, t + kPrefillTile));
                    tile_seq_.push_back(s);
                }
            } else {
                for (int t = q0; t < q1; t++) decode_tokens_.push_back(t);
                max_ctx = std::max(max_ctx, b.context_len[static_cast<size_t>(s)]);
            }
        }
        Plan p;
        p.T = T;
        p.R = R;
        p.max_blocks = b.max_blocks;
        p.decode_count = static_cast<int>(decode_tokens_.size());
        p.tiles = static_cast<int>(tile_start_.size());
        p.tokens = put(b.tokens.data(), static_cast<size_t>(T));
        p.positions = put(b.positions.data(), static_cast<size_t>(T));
        p.token_seq = put(token_seq_.data(), static_cast<size_t>(T));
        p.tables = put(b.block_tables.data(), b.block_tables.size());
        p.rows = put(b.logit_rows.data(), static_cast<size_t>(R));
        p.decode = put(decode_tokens_.data(), decode_tokens_.size());
        p.tile_start = put(tile_start_.data(), tile_start_.size());
        p.tile_end = put(tile_end_.data(), tile_end_.size());
        p.tile_seq = put(tile_seq_.data(), tile_seq_.size());
        choose_splits(p, max_ctx);
        EMBER_CHECK(off <= host_ints_count_, "internal: step buffer too small");
        CUDA_CHECK(cudaMemcpyAsync(dev_ints_.get(), hp, off * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        enqueue(p);
        CUDA_CHECK(cudaGetLastError());
    }

    void sample(std::span<const SampleRequest> reqs, std::span<int32_t> out) override {
        EMBER_CHECK(static_cast<int>(reqs.size()) == rows_ && out.size() == reqs.size(), "sample: {} requests for {} rows",
                    reqs.size(), rows_);
        if (reqs.empty()) return;
        upload_params(reqs);
        sample_tokens(logits_.get(), rows_, cfg_.vocab_size, dev_params_.get(), dev_tokens_.get(), stream_);
        CUDA_CHECK(cudaMemcpyAsync(host_tokens_, dev_tokens_.get(), reqs.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::memcpy(out.data(), host_tokens_, reqs.size() * sizeof(int32_t));
    }

    void token_probs(std::span<const SampleRequest> reqs, std::span<const int32_t> rows, std::span<const int32_t> tokens,
                     std::span<float> out) override {
        EMBER_CHECK(static_cast<int>(reqs.size()) == rows_, "token_probs: params must cover every logit row");
        const size_t n = rows.size();
        if (n == 0) return;
        EMBER_CHECK(n <= static_cast<size_t>(opt_.max_logit_rows), "token_probs: too many queries");
        upload_params(reqs);
        std::vector<int32_t> packed(2 * n);
        std::copy(rows.begin(), rows.end(), packed.begin());
        std::copy(tokens.begin(), tokens.end(), packed.begin() + static_cast<ptrdiff_t>(n));
        CUDA_CHECK(cudaMemcpyAsync(dev_queries_.get(), packed.data(), packed.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        token_probabilities(logits_.get(), cfg_.vocab_size, dev_params_.get(), dev_queries_.get(),
                            dev_queries_.get() + n, static_cast<int>(n), dev_probs_.get(), stream_);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), dev_probs_.get(), n * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void logits(int row, std::span<float> out) override {
        EMBER_CHECK(row >= 0 && row < rows_ && static_cast<int>(out.size()) >= cfg_.vocab_size, "logits: bad row {}", row);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_.get() + static_cast<size_t>(row) * cfg_.vocab_size,
                                   sizeof(float) * cfg_.vocab_size, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void set_capture_hidden(bool on) override {
        capture_ = on;
        if (on && !captured_.size()) captured_.resize(static_cast<size_t>(cfg_.n_layers + 1) * cfg_.hidden);
    }

    std::vector<float> captured_hidden() override {
        std::vector<float> out(captured_.size());
        CUDA_CHECK(cudaMemcpyAsync(out.data(), captured_.get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return out;
    }

private:
    // Decode attention splits long contexts into fixed chunks of kSplitLen keys.
    static constexpr int kSplitLen = 512, kMaxSplits = 80, kMaxSplitTokens = 64;
    static constexpr int kFlashMinTokens = 16;  // a sequence with this many new tokens uses the prefill kernel

    static double mib(size_t b) { return static_cast<double>(b) / (1 << 20); }

    // Device-side description of one step: pointers into the uploaded step
    // buffer, and the work split. Everything a kernel launch needs.
    struct Plan {
        int T = 0, R = 0, max_blocks = 0, decode_count = 0, tiles = 0;
        int splits = 1, split_len = kBlockSize;
        const int32_t *tokens = nullptr, *positions = nullptr, *token_seq = nullptr, *tables = nullptr, *rows = nullptr,
                      *decode = nullptr, *tile_start = nullptr, *tile_end = nullptr, *tile_seq = nullptr;
    };

    static constexpr int kGraphMaxTokens = 64;

    // Decode attention split. The chunks are fixed (keys [0, 512), [512, 1024), ...)
    // and merged in a fixed order, so a token's attention is computed the same way
    // whatever else is in the batch: decoding is batch-invariant (a request yields
    // the same tokens alone or among others, and speculative decoding reproduces
    // plain decoding exactly). Only beyond 64 decode tokens per step, or contexts
    // over 40k, does it fall back to one pass per (token, head).
    void choose_splits(Plan &p, int max_ctx) const {
        p.split_len = kSplitLen;
        p.splits = std::max(1, (max_ctx + kSplitLen - 1) / kSplitLen);
        if (p.decode_count > kMaxSplitTokens || p.splits > kMaxSplits) {
            p.splits = 1;
            p.split_len = std::max(kBlockSize, (max_ctx + kBlockSize - 1) / kBlockSize * kBlockSize);
        }
    }

    // Enqueues the whole forward pass for a plan on stream_.
    void enqueue(const Plan &p) {
        const ModelConfig &c = cfg_;
        const int T = p.T, H = c.hidden, Q = c.q_dim(), KV = c.kv_dim(), I = c.intermediate;

        if (lm_head_.qt != QuantType::none && embed_ == nullptr)
            embed_tokens_quant(p.tokens, T, lm_head_.wq, lm_head_.scales, lm_head_.qt, H, h_.get(), stream_);
        else
            embed_tokens(p.tokens, T, embed_, H, h_.get(), stream_);
        capture(0, T);

        AttentionArgs att{};
        att.token_index = p.decode;
        att.q = q_.get();
        att.positions = p.positions;
        att.token_seq = p.token_seq;
        att.block_tables = p.tables;
        att.max_blocks = p.max_blocks;
        att.T = p.decode_count;
        att.n_heads = c.n_heads;
        att.n_kv = c.n_kv_heads;
        att.head_dim = c.head_dim;
        att.scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
        att.out = att_.get();
        att.num_splits = p.splits;
        att.split_len = p.split_len;
        att.part_acc = part_acc_.get();
        att.part_ml = part_ml_.get();

        PrefillArgs pre{};
        pre.q = q_.get();
        pre.tile_start = p.tile_start;
        pre.tile_end = p.tile_end;
        pre.tile_seq = p.tile_seq;
        pre.positions = p.positions;
        pre.block_tables = p.tables;
        pre.num_tiles = p.tiles;
        pre.max_blocks = p.max_blocks;
        pre.n_heads = c.n_heads;
        pre.n_kv = c.n_kv_heads;
        pre.head_dim = c.head_dim;
        pre.scale = att.scale;
        pre.out = att_.get();

        QkvArgs qa{};
        qa.qkv = qkv_.get();
        qa.q_out = q_.get();
        qa.inv_freq = inv_freq_.get();
        qa.positions = p.positions;
        qa.token_seq = p.token_seq;
        qa.block_tables = p.tables;
        qa.max_blocks = p.max_blocks;
        qa.T = T;
        qa.n_heads = c.n_heads;
        qa.n_kv = c.n_kv_heads;
        qa.head_dim = c.head_dim;
        qa.eps = c.rms_eps;

        for (int l = 0; l < c.n_layers; l++) {
            const LayerWeights &L = layers_[static_cast<size_t>(l)];
            half *kc = k_cache_.get() + static_cast<size_t>(l) * layer_cache_elems_;
            half *vc = v_cache_.get() + static_cast<size_t>(l) * layer_cache_elems_;

            rmsnorm_rows(h_.get(), nullptr, T, H, L.in_norm, c.rms_eps, xn_.get(), stream_);
            linear(L.qkv, xn_.get(), H, T, qkv_.get(), Q + 2 * KV, Epilogue::store_f16);
            qa.q_norm = L.q_norm;
            qa.k_norm = L.k_norm;
            qa.bias = L.qkv_bias;
            qa.k_cache = kc;
            qa.v_cache = vc;
            qk_norm_rope_store(qa, stream_);
            att.k_cache = pre.k_cache = kc;
            att.v_cache = pre.v_cache = vc;
            if (p.decode_count > 0) paged_attention(att, stream_);
            if (p.tiles > 0) prefill_attention(pre, stream_);
            linear(L.o, att_.get(), Q, T, h_.get(), H, Epilogue::add_f32);

            rmsnorm_rows(h_.get(), nullptr, T, H, L.post_norm, c.rms_eps, xn_.get(), stream_);
            linear(L.gate_up, xn_.get(), H, T, act_.get(), I, Epilogue::silu_mul);
            linear(L.down, act_.get(), I, T, h_.get(), H, Epilogue::add_f32);
            capture(l + 1, T);
        }

        if (p.R > 0) {
            rmsnorm_rows(h_.get(), p.rows, p.R, H, final_norm_, c.rms_eps, xn_.get(), stream_);
            linear(lm_head_, xn_.get(), H, p.R, logits_.get(), c.vocab_size, Epilogue::store_f32);
        }
    }

    // ---- CUDA graphs for decode steps ------------------------------------------------
    //
    // A graph freezes kernel arguments, so the step buffer has a fixed layout per
    // (token bucket, block-table width bucket), and the batch is padded up to the
    // bucket: padding tokens are sequences of length 1 that write into a spare KV
    // block the engine never hands out. The attention split depends only on the
    // buckets, so a replay always matches its capture.

    static int token_bucket(int T) {
        static const int buckets[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64};
        for (int b : buckets)
            if (T <= b) return b;
        return kGraphMaxTokens;
    }

    int width_bucket(int w) const {
        int b = 16;
        while (b < w) b *= 4;
        return std::min(b, max_blocks_per_seq_);
    }

    struct GraphEntry {
        cudaGraphExec_t exec = nullptr;
        Plan plan;
    };

    void forward_graph(const StepBatch &b) {
        const int T = b.num_tokens();
        const int TB = token_bucket(T), WB = width_bucket(std::max(b.max_blocks, 1));

        // Fixed layout: tokens, positions, sequence ids, rows (identity), tables [TB x WB].
        int32_t *hp = host_ints_;
        const size_t pad = (static_cast<size_t>(TB) + 31) & ~size_t(31);
        int32_t *tok = hp, *pos = hp + pad, *seq = hp + 2 * pad, *rows = hp + 3 * pad, *tab = hp + 4 * pad;
        for (int t = 0; t < TB; t++) {
            const bool real = t < T;
            tok[t] = real ? b.tokens[static_cast<size_t>(t)] : 0;
            pos[t] = real ? b.positions[static_cast<size_t>(t)] : 0;
            seq[t] = t;
            rows[t] = t;
            int32_t *row = tab + static_cast<size_t>(t) * WB;
            if (real) {
                std::copy_n(b.block_tables.begin() + static_cast<ptrdiff_t>(t) * b.max_blocks, b.max_blocks, row);
                std::fill(row + b.max_blocks, row + WB, spare_block_);
            } else {
                std::fill(row, row + WB, spare_block_);
            }
        }
        const size_t used = 4 * pad + static_cast<size_t>(TB) * WB;
        EMBER_CHECK(used <= host_ints_count_, "internal: step buffer too small for a graph");
        CUDA_CHECK(cudaMemcpyAsync(dev_ints_.get(), hp, used * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

        const uint64_t key = (static_cast<uint64_t>(TB) << 32) | static_cast<uint32_t>(WB);
        auto it = graphs_.find(key);
        if (it == graphs_.end()) {
            GraphEntry e;
            Plan &p = e.plan;
            p.T = p.R = p.decode_count = TB;
            p.max_blocks = WB;
            p.tiles = 0;
            int32_t *d = dev_ints_.get();
            p.tokens = d;
            p.positions = d + pad;
            p.token_seq = d + 2 * pad;
            p.rows = d + 3 * pad;
            p.tables = d + 4 * pad;
            p.decode = nullptr;  // identity: every token is a decode token
            choose_splits(p, WB * kBlockSize);
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            enqueue(p);
            CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&e.exec, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            it = graphs_.emplace(key, e).first;
            log::debug("captured a decode graph for {} tokens x {} blocks", TB, WB);
        }
        CUDA_CHECK(cudaGraphLaunch(it->second.exec, stream_));
    }

    void linear(const Linear &L, const half *A, int lda, int M, void *C, int ldc, Epilogue ep) {
        if (L.qt == QuantType::none) {
            gemm(A, lda, L.w, M, L.N, L.K, C, ldc, ep, stream_, GemmKernel::automatic, &gemm_ws_);
            return;
        }
        // The quantized decode kernels take up to 64 rows; the prefill path dequantizes
        // into a scratch sized for the layer matrices, never the LM head - so the LM
        // head always goes 64 rows at a time.
        const bool big = static_cast<size_t>(L.N) * L.K > quant_scratch_.dequant_elems;
        const int step = big ? 64 : M;
        for (int m0 = 0; m0 < M; m0 += step) {
            const int rows = std::min(step, M - m0);
            const size_t out_elem = ep == Epilogue::store_f32 || ep == Epilogue::add_f32 ? sizeof(float) : sizeof(half);
            void *Cm = static_cast<char *>(C) + static_cast<size_t>(m0) * ldc * out_elem;
            gemm_quant(A + static_cast<size_t>(m0) * lda, lda, L.wq, L.scales, L.qt, rows, L.N, L.K, Cm, ldc, ep,
                       quant_scratch_, stream_);
        }
    }

    void capture(int index, int T) {
        if (!capture_) return;
        const int H = cfg_.hidden;
        CUDA_CHECK(cudaMemcpyAsync(captured_.get() + static_cast<size_t>(index) * H, h_.get() + static_cast<size_t>(T - 1) * H,
                                   H * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    }

    void upload_params(std::span<const SampleRequest> reqs) {
        for (size_t i = 0; i < reqs.size(); i++) {
            const SamplingParams &p = reqs[i].params;
            host_params_[i] = {p.temperature, p.top_p, p.min_p, p.top_k, p.seed, reqs[i].counter};
        }
        CUDA_CHECK(cudaMemcpyAsync(dev_params_.get(), host_params_, reqs.size() * sizeof(DeviceSampleParams),
                                   cudaMemcpyHostToDevice, stream_));
    }

    // ---- loading -------------------------------------------------------------------

    // Uploads rows of a tensor through the staging buffer and converts them to f16
    // at destination rows dst_offset + r * dst_stride.
    void upload_rows(const TensorView &t, half *dst, int64_t dst_offset, int64_t dst_stride) {
        const int64_t rows = t.shape.size() == 1 ? 1 : t.shape[0];
        const int64_t cols = t.shape.size() == 1 ? t.shape[0] : t.shape[1];
        const size_t elem = dtype_size(t.dtype);
        const int64_t rows_per_chunk = std::max<int64_t>(1, static_cast<int64_t>(staging_.size() / (cols * elem)));
        const uint8_t *src = static_cast<const uint8_t *>(t.data);
        for (int64_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
            int64_t n = std::min(rows_per_chunk, rows - r0);
            // Same stream as the conversion: a pageable cudaMemcpy may return before
            // its DMA completes, and a kernel on another stream would race it.
            CUDA_CHECK(cudaMemcpyAsync(staging_.get(), src + static_cast<size_t>(r0 * cols) * elem,
                                       static_cast<size_t>(n * cols) * elem, cudaMemcpyHostToDevice, stream_));
            convert_rows_to_f16(staging_.get(), src_type(t.dtype), n, cols, dst, dst_offset + r0 * dst_stride, dst_stride,
                                overflow_.get(), stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));  // the staging buffer is reused
        }
    }

    float *upload_vector(const TensorView &t) {
        float *dst = arena_.take<float>(static_cast<size_t>(t.numel()));
        CUDA_CHECK(cudaMemcpyAsync(staging_.get(), t.data, t.bytes, cudaMemcpyHostToDevice, stream_));
        convert_to_f32(staging_.get(), src_type(t.dtype), t.numel(), dst, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return dst;
    }

    // Bytes a linear layer takes in the arena, in the chosen format.
    size_t linear_bytes(int64_t N, int64_t K) const {
        switch (qtype_) {
            case QuantType::none: return aligned(static_cast<size_t>(N * K) * 2);
            case QuantType::int8: return aligned(static_cast<size_t>(N * K)) + aligned(static_cast<size_t>(N) * 4);
            case QuantType::int4: return aligned(static_cast<size_t>(N * K / 2)) + aligned(static_cast<size_t>(N * (K / kInt4Group)) * 4);
        }
        return 0;
    }

    // Allocates a linear layer. With quantization, its f16 rows are first
    // assembled in `temp_` (by upload_rows), then finish_linear() packs them.
    // The format for one kind of matrix: --weights, unless --quant-mix overrides it.
    // The int4 recipe keeps the matrices most sensitive to 4 bits in int8: the
    // q/k/v projections and the MLP down projection (measured with `ember
    // perplexity`: pure int4 costs +50% perplexity on Qwen3-1.7B, this mix ~0-5%).
    QuantType format_for(const std::string &kind, QuantType dflt) const {
        if (dflt == QuantType::int4 && (kind == "qkv" || kind == "down")) dflt = QuantType::int8;
        const std::string &mix = opt_.quant_mix;
        for (size_t p = 0; p < mix.size();) {
            size_t q = mix.find(',', p);
            std::string item = mix.substr(p, q == std::string::npos ? std::string::npos : q - p);
            p = q == std::string::npos ? mix.size() : q + 1;
            size_t eq = item.find('=');
            if (eq == std::string::npos) fail("--quant-mix: expected kind=format, got \"{}\"", item);
            if (item.substr(0, eq) != kind) continue;
            WeightFormat f = parse_weight_format(item.substr(eq + 1));
            return f == WeightFormat::int8 ? QuantType::int8 : f == WeightFormat::int4 ? QuantType::int4 : QuantType::none;
        }
        return dflt;
    }

    Linear make_linear(int N, int K, const std::string &kind) {
        const QuantType saved = qtype_;
        qtype_ = format_for(kind, qtype_);
        Linear L = make_linear(N, K);
        qtype_ = saved;
        return L;
    }

    Linear make_linear(int N, int K) {
        Linear L;
        L.N = N;
        L.K = K;
        if (qtype_ == QuantType::none) {
            L.w = arena_.take<half>(static_cast<size_t>(N) * K);
            return L;
        }
        EMBER_CHECK(K % kInt4Group == 0, "quantization needs layer widths divisible by 128 (got {})", K);
        L.qt = qtype_;
        if (qtype_ == QuantType::int8) {
            L.wq = arena_.take<int8_t>(static_cast<size_t>(N) * K);
            L.scales = reinterpret_cast<half *>(arena_.take<float>(static_cast<size_t>(N)));
        } else {
            L.wq = arena_.take<uint8_t>(static_cast<size_t>(N) * K / 2);
            L.scales = reinterpret_cast<half *>(arena_.take<uint32_t>(static_cast<size_t>(N) * (K / kInt4Group)));
        }
        L.w = temp_.get();  // staging target for upload_rows
        return L;
    }

    void finish_linear(Linear &L) {
        if (L.qt == QuantType::none) {
            if (L.w == temp_.get()) fail("internal: f16 linear staged in the temp buffer");
            return;
        }
        quantize_weights(temp_.get(), L.N, L.K, L.qt, L.wq, L.scales, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        L.w = nullptr;
    }

    void load_weights(const std::string &dir) {
        SafeTensors st = SafeTensors::open_dir(dir);
        const ModelConfig &c = cfg_;
        const int64_t H = c.hidden, Q = c.q_dim(), KV = c.kv_dim(), I = c.intermediate, D = c.head_dim, V = c.vocab_size;
        const bool tied = c.tie_embeddings || !st.has("lm_head.weight");

        // Size the arena: matrices in the chosen format, vectors in f32 (rounded per allocation).
        // With quantization the tied embedding is stored once, quantized, and serves both uses.
        const bool quant = qtype_ != QuantType::none;
        size_t head_bytes = aligned(static_cast<size_t>(V * H)) + aligned(static_cast<size_t>(V) * 4);  // int8 output layer
        if (format_for("lm_head", QuantType::int8) == QuantType::int4)
            head_bytes = aligned(static_cast<size_t>(V * H / 2)) + aligned(static_cast<size_t>(V * (H / kInt4Group)) * 4);
        size_t bytes = (quant ? head_bytes + (tied ? 0 : aligned(static_cast<size_t>(V * H) * 2))
                              : aligned(static_cast<size_t>(V * H) * 2) * (tied ? 1 : 2)) +
                       aligned(static_cast<size_t>(H) * 4);
        auto kind_bytes = [&](const char *kind, int64_t n, int64_t k) {
            const QuantType saved = qtype_;
            qtype_ = format_for(kind, qtype_);
            size_t b = linear_bytes(n, k);
            qtype_ = saved;
            return b;
        };
        size_t per_layer = kind_bytes("qkv", Q + 2 * KV, H) + kind_bytes("o", H, Q) + kind_bytes("gate_up", 2 * I, H) +
                           kind_bytes("down", H, I) +
                           2 * aligned(static_cast<size_t>(H) * 4) + 2 * aligned(static_cast<size_t>(D) * 4) +
                           aligned(static_cast<size_t>(Q + 2 * KV) * 4);
        bytes += per_layer * static_cast<size_t>(c.n_layers) + aligned(static_cast<size_t>(D / 2) * 4);
        size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        EMBER_CHECK(bytes + (256u << 20) < free, "the model needs {:.0f} MiB of GPU memory but only {:.0f} MiB is free",
                    mib(bytes), mib(free));
        arena_.reserve(bytes);
        staging_.resize(64u << 20);
        if (quant) temp_.resize(static_cast<size_t>(std::max({V * H, (Q + 2 * KV) * H, 2 * I * H, H * I, H * Q})));
        overflow_.resize(1);
        CUDA_CHECK(cudaMemsetAsync(overflow_.get(), 0, sizeof(unsigned long long), stream_));

        if (quant) {
            // Quantized: the LM head is quantized; a tied embedding reads from it.
            if (!tied) {
                embed_ = arena_.take<half>(static_cast<size_t>(V * H));
                upload_rows(st.get("model.embed_tokens.weight", {V, H}), embed_, 0, 1);
            }
            // The output layer is the most sensitive to quantization: with int4 it keeps 8 bits
            // (as llama.cpp's Q4_K_M keeps more bits there).
            const QuantType body = qtype_;
            qtype_ = format_for("lm_head", qtype_ == QuantType::int4 ? QuantType::int8 : qtype_);
            EMBER_CHECK(qtype_ != QuantType::none, "--quant-mix: lm_head=f16 is not supported with quantized weights");
            lm_head_ = make_linear(static_cast<int>(V), static_cast<int>(H));
            upload_rows(st.get(tied ? "model.embed_tokens.weight" : "lm_head.weight", {V, H}), lm_head_.w, 0, 1);
            finish_linear(lm_head_);
            qtype_ = body;
        } else {
            embed_ = arena_.take<half>(static_cast<size_t>(V * H));
            upload_rows(st.get("model.embed_tokens.weight", {V, H}), embed_, 0, 1);
            lm_head_.N = static_cast<int>(V);
            lm_head_.K = static_cast<int>(H);
            if (tied) {
                lm_head_.w = embed_;
            } else {
                lm_head_.w = arena_.take<half>(static_cast<size_t>(V * H));
                upload_rows(st.get("lm_head.weight", {V, H}), lm_head_.w, 0, 1);
            }
        }
        final_norm_ = upload_vector(st.get("model.norm.weight", {H}));

        layers_.resize(static_cast<size_t>(c.n_layers));
        for (int l = 0; l < c.n_layers; l++) {
            const std::string p = std::format("model.layers.{}.", l);
            LayerWeights &L = layers_[static_cast<size_t>(l)];
            L.qkv = make_linear(static_cast<int>(Q + 2 * KV), static_cast<int>(H), "qkv");
            upload_rows(st.get(p + "self_attn.q_proj.weight", {Q, H}), L.qkv.w, 0, 1);
            upload_rows(st.get(p + "self_attn.k_proj.weight", {KV, H}), L.qkv.w, Q, 1);
            upload_rows(st.get(p + "self_attn.v_proj.weight", {KV, H}), L.qkv.w, Q + KV, 1);
            finish_linear(L.qkv);
            L.o = make_linear(static_cast<int>(H), static_cast<int>(Q), "o");
            upload_rows(st.get(p + "self_attn.o_proj.weight", {H, Q}), L.o.w, 0, 1);
            finish_linear(L.o);
            L.gate_up = make_linear(static_cast<int>(2 * I), static_cast<int>(H), "gate_up");
            upload_rows(st.get(p + "mlp.gate_proj.weight", {I, H}), L.gate_up.w, 0, 2);  // even rows
            upload_rows(st.get(p + "mlp.up_proj.weight", {I, H}), L.gate_up.w, 1, 2);    // odd rows
            finish_linear(L.gate_up);
            L.down = make_linear(static_cast<int>(H), static_cast<int>(I), "down");
            upload_rows(st.get(p + "mlp.down_proj.weight", {H, I}), L.down.w, 0, 1);
            finish_linear(L.down);
            L.in_norm = upload_vector(st.get(p + "input_layernorm.weight", {H}));
            L.post_norm = upload_vector(st.get(p + "post_attention_layernorm.weight", {H}));
            if (c.qk_norm) {
                L.q_norm = upload_vector(st.get(p + "self_attn.q_norm.weight", {D}));
                L.k_norm = upload_vector(st.get(p + "self_attn.k_norm.weight", {D}));
            }
            if (c.qkv_bias) {
                L.qkv_bias = arena_.take<float>(static_cast<size_t>(Q + 2 * KV));
                const TensorView *bs[3] = {&st.get(p + "self_attn.q_proj.bias", {Q}), &st.get(p + "self_attn.k_proj.bias", {KV}),
                                           &st.get(p + "self_attn.v_proj.bias", {KV})};
                int64_t at = 0;
                for (const TensorView *bt : bs) {
                    CUDA_CHECK(cudaMemcpyAsync(staging_.get(), bt->data, bt->bytes, cudaMemcpyHostToDevice, stream_));
                    convert_to_f32(staging_.get(), src_type(bt->dtype), bt->numel(), L.qkv_bias + at, stream_);
                    CUDA_CHECK(cudaStreamSynchronize(stream_));
                    at += bt->numel();
                }
            }
        }

        std::vector<float> inv(static_cast<size_t>(D / 2));
        for (int i = 0; i < D / 2; i++)  // float32, as transformers computes it
            inv[static_cast<size_t>(i)] =
                1.0f / std::pow(static_cast<float>(c.rope_theta), static_cast<float>(2 * i) / static_cast<float>(D));
        inv_freq_.resize(inv.size());
        CUDA_CHECK(cudaMemcpyAsync(inv_freq_.get(), inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        unsigned long long overflow = 0;
        CUDA_CHECK(cudaMemcpyAsync(&overflow, overflow_.get(), sizeof overflow, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (overflow) log::warn("{} weights were outside the f16 range and were clamped", overflow);
        staging_.release();
        temp_.release();
        weight_bytes_ = arena_.used();
    }

    void allocate_activations() {
        const ModelConfig &c = cfg_;
        const size_t T = static_cast<size_t>(opt_.max_batch_tokens), H = c.hidden, Q = c.q_dim(), KV = c.kv_dim(),
                     I = c.intermediate;
        const size_t R = static_cast<size_t>(opt_.max_logit_rows);
        h_.resize(T * H);
        xn_.resize(std::max(T, R) * H);
        qkv_.resize(T * (Q + 2 * KV));
        q_.resize(T * Q);
        att_.resize(T * Q);
        act_.resize(T * I);
        logits_.resize(R * c.vocab_size);
        const size_t split_tokens = std::min<size_t>(T, kMaxSplitTokens);
        part_acc_.resize(split_tokens * c.n_heads * kMaxSplits * c.head_dim);
        part_ml_.resize(split_tokens * c.n_heads * kMaxSplits * 2);
        // Split-K partials: up to 8 splits of a [64, max N] product (decode-sized batches).
        split_partial_.resize(8ull * 64 * std::max<size_t>({Q + 2 * KV, 2 * I, H}));
        gemm_ws_.partial = split_partial_.get();
        gemm_ws_.partial_elems = split_partial_.size();
        if (qtype_ != QuantType::none) {
            // Prefill dequantizes one layer matrix at a time into f16.
            quant_dequant_.resize(std::max({(Q + 2 * KV) * H, H * Q, 2 * I * H, H * I}));
            quant_xsum_.resize(64 * (std::max({H, Q, I}) / kInt4Group + 1));
            quant_scratch_ = {quant_dequant_.get(), quant_dequant_.size(), quant_xsum_.get(), quant_xsum_.size(), &gemm_ws_};
        }

        max_blocks_per_seq_ = (c.max_position + kBlockSize - 1) / kBlockSize;
        // tokens, positions, sequence ids, decode list, 3 x tiles, block tables, logit rows (each padded to 32)
        host_ints_count_ = 4 * (T + 32) + 3 * (T / 16 + 32) + static_cast<size_t>(opt_.max_seqs) * max_blocks_per_seq_ + 32 + R + 32;
        dev_ints_.resize(host_ints_count_);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_ints_), host_ints_count_ * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_params_), R * sizeof(DeviceSampleParams)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_tokens_), R * sizeof(int32_t)));
        dev_params_.resize(R);
        dev_tokens_.resize(R);
        dev_queries_.resize(2 * R);
        dev_probs_.resize(R);

        act_bytes_ = h_.bytes() + xn_.bytes() + qkv_.bytes() + q_.bytes() + att_.bytes() + act_.bytes() + logits_.bytes() +
                     part_acc_.bytes() + part_ml_.bytes() + dev_ints_.bytes() + split_partial_.bytes() + quant_dequant_.bytes() +
                     quant_xsum_.bytes();
    }

    void allocate_kv_cache() {
        const ModelConfig &c = cfg_;
        layer_block_elems_ = static_cast<size_t>(c.n_kv_heads) * kBlockSize * c.head_dim;
        const size_t block_bytes = 2 * layer_block_elems_ * sizeof(half) * c.n_layers;  // K and V, all layers
        size_t bytes;
        if (opt_.kv_cache_bytes > 0) {
            bytes = static_cast<size_t>(opt_.kv_cache_bytes);
        } else {
            size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            const size_t keep = static_cast<size_t>((1.0 - opt_.memory_fraction) * static_cast<double>(total));
            bytes = free > keep + (64u << 20) ? free - keep - (64u << 20) : 0;
        }
        // One extra block, never handed to the engine: the padding tokens of decode graphs write there.
        num_blocks_ = static_cast<int>(bytes / block_bytes) - 1;
        EMBER_CHECK(num_blocks_ >= 4, "not enough GPU memory left for the KV cache ({:.0f} MiB)", mib(bytes));
        spare_block_ = num_blocks_;
        layer_cache_elems_ = layer_block_elems_ * (num_blocks_ + 1);
        k_cache_.resize(layer_cache_elems_ * c.n_layers);
        v_cache_.resize(layer_cache_elems_ * c.n_layers);
        CUDA_CHECK(cudaMemset(k_cache_.get(), 0, k_cache_.bytes()));  // unwritten slots read as zeros, never NaN
        CUDA_CHECK(cudaMemset(v_cache_.get(), 0, v_cache_.bytes()));
        kv_bytes_ = k_cache_.bytes() + v_cache_.bytes();
    }

    ModelConfig cfg_;
    BackendOptions opt_;
    cudaDeviceProp props_{};
    cudaStream_t stream_ = nullptr;

    Arena arena_;
    QuantType qtype_ = QuantType::none;
    DeviceBuffer<uint8_t> staging_;
    DeviceBuffer<half> temp_, quant_dequant_;
    DeviceBuffer<float> quant_xsum_;
    QuantScratch quant_scratch_;
    DeviceBuffer<unsigned long long> overflow_;
    half *embed_ = nullptr;
    Linear lm_head_;
    float *final_norm_ = nullptr;
    std::vector<LayerWeights> layers_;
    DeviceBuffer<float> inv_freq_;
    size_t weight_bytes_ = 0;

    int num_blocks_ = 0;
    size_t layer_block_elems_ = 0, layer_cache_elems_ = 0, kv_bytes_ = 0;
    DeviceBuffer<half> k_cache_, v_cache_;  // [layer][block][kv_head][slot][head_dim]

    DeviceBuffer<float> h_, logits_, part_acc_, part_ml_, captured_, split_partial_;
    GemmWorkspace gemm_ws_;
    DeviceBuffer<half> xn_, qkv_, q_, att_, act_;
    size_t act_bytes_ = 0;

    int max_blocks_per_seq_ = 0;
    size_t host_ints_count_ = 0;
    int32_t *host_ints_ = nullptr;
    DeviceBuffer<int32_t> dev_ints_;
    DeviceSampleParams *host_params_ = nullptr;
    int32_t *host_tokens_ = nullptr;
    DeviceBuffer<DeviceSampleParams> dev_params_;
    DeviceBuffer<int32_t> dev_tokens_, dev_queries_;
    DeviceBuffer<float> dev_probs_;

    std::vector<int32_t> token_seq_, decode_tokens_, tile_start_, tile_end_, tile_seq_;
    std::unordered_map<uint64_t, GraphEntry> graphs_;
    int spare_block_ = 0;
    int rows_ = 0;
    bool capture_ = false;
};

}  // namespace

// ---- test hooks ---------------------------------------------------------------------

std::vector<float> test_gemm(const std::vector<uint16_t> &A, const std::vector<uint16_t> &W, int M, int N, int K,
                             int epilogue, int kernel, bool split_k) {
    DeviceBuffer<half> dA(A.size()), dW(W.size());
    DeviceBuffer<float> partial(split_k ? 8ull * M * N : 0);
    GemmWorkspace workspace{partial.get(), partial.size()};
    const GemmWorkspace *ws = split_k ? &workspace : nullptr;
    CUDA_CHECK(cudaMemcpy(dA.get(), A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW.get(), W.data(), W.size() * 2, cudaMemcpyHostToDevice));
    auto ep = static_cast<Epilogue>(epilogue);
    const int out_cols = ep == Epilogue::silu_mul ? N / 2 : N;
    std::vector<float> out(static_cast<size_t>(M) * out_cols, 0.0f);
    if (ep == Epilogue::store_f32 || ep == Epilogue::add_f32) {
        DeviceBuffer<float> dC(out.size());
        // add_f32 adds to an initial 1.0 everywhere, so the test sees the accumulation.
        std::vector<float> init(out.size(), ep == Epilogue::add_f32 ? 1.0f : 0.0f);
        CUDA_CHECK(cudaMemcpy(dC.get(), init.data(), init.size() * 4, cudaMemcpyHostToDevice));
        gemm(dA.get(), K, dW.get(), M, N, K, dC.get(), out_cols, ep, nullptr, static_cast<GemmKernel>(kernel), ws);
        CUDA_CHECK(cudaMemcpy(out.data(), dC.get(), out.size() * 4, cudaMemcpyDeviceToHost));
    } else {
        DeviceBuffer<half> dC(out.size());
        gemm(dA.get(), K, dW.get(), M, N, K, dC.get(), out_cols, ep, nullptr, static_cast<GemmKernel>(kernel), ws);
        std::vector<uint16_t> h(out.size());
        CUDA_CHECK(cudaMemcpy(h.data(), dC.get(), h.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < h.size(); i++) {
            half v;
            std::memcpy(&v, &h[i], 2);
            out[i] = __half2float(v);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

double bench_gemm(int M, int N, int K, int epilogue, int kernel, bool split_k, int iters) {
    DeviceBuffer<half> dA(static_cast<size_t>(M) * K), dW(static_cast<size_t>(N) * K);
    DeviceBuffer<float> dC(static_cast<size_t>(M) * N);
    DeviceBuffer<float> partial(split_k ? 8ull * M * N : 0);
    CUDA_CHECK(cudaMemset(dA.get(), 0, dA.bytes()));
    CUDA_CHECK(cudaMemset(dW.get(), 0, dW.bytes()));
    CUDA_CHECK(cudaMemset(dC.get(), 0, dC.bytes()));
    GemmWorkspace workspace{partial.get(), partial.size()};
    auto ep = static_cast<Epilogue>(epilogue);
    const int ldc = ep == Epilogue::silu_mul ? N / 2 : N;
    auto run = [&] {
        gemm(dA.get(), K, dW.get(), M, N, K, dC.get(), ldc, ep, nullptr, static_cast<GemmKernel>(kernel),
             split_k ? &workspace : nullptr);
    };
    run();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) run();
    cudaEventRecord(e1);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms * 1000.0 / iters;
}

QuantTestResult test_quant_gemm(const std::vector<uint16_t> &A, const std::vector<uint16_t> &W, int M, int N, int K, int qt,
                                int epilogue) {
    const QuantType q = static_cast<QuantType>(qt);
    DeviceBuffer<half> dA(A.size()), dW(W.size()), deq(W.size());
    CUDA_CHECK(cudaMemcpy(dA.get(), A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW.get(), W.data(), W.size() * 2, cudaMemcpyHostToDevice));
    DeviceBuffer<uint8_t> wq(static_cast<size_t>(N) * K);
    DeviceBuffer<uint32_t> sc(static_cast<size_t>(N) * (K / kInt4Group + 1));
    quantize_weights(dW.get(), N, K, q, wq.get(), reinterpret_cast<half *>(sc.get()), nullptr);
    dequantize_weights(wq.get(), reinterpret_cast<half *>(sc.get()), q, N, K, deq.get(), nullptr);

    auto ep = static_cast<Epilogue>(epilogue);
    const int cols = ep == Epilogue::silu_mul ? N / 2 : N;
    const bool f32 = ep == Epilogue::store_f32 || ep == Epilogue::add_f32;
    DeviceBuffer<float> xsum(static_cast<size_t>(M) * (K / kInt4Group));
    QuantScratch scratch{nullptr, 0, xsum.get(), xsum.size(), nullptr};
    auto run = [&](bool quant) {
        std::vector<float> out(static_cast<size_t>(M) * cols);
        DeviceBuffer<uint8_t> dC(out.size() * 4);
        CUDA_CHECK(cudaMemset(dC.get(), 0, dC.bytes()));
        if (quant) gemm_quant(dA.get(), K, wq.get(), reinterpret_cast<half *>(sc.get()), q, M, N, K, dC.get(), cols, ep, scratch, nullptr);
        else gemm(dA.get(), K, deq.get(), M, N, K, dC.get(), cols, ep, nullptr, GemmKernel::automatic, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (f32) {
            CUDA_CHECK(cudaMemcpy(out.data(), dC.get(), out.size() * 4, cudaMemcpyDeviceToHost));
        } else {
            std::vector<half> h(out.size());
            CUDA_CHECK(cudaMemcpy(h.data(), dC.get(), h.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < h.size(); i++) out[i] = __half2float(h[i]);
        }
        return out;
    };
    QuantTestResult r;
    r.quant = run(true);
    r.reference = run(false);
    std::vector<half> hw(W.size());
    CUDA_CHECK(cudaMemcpy(hw.data(), deq.get(), hw.size() * 2, cudaMemcpyDeviceToHost));
    r.dequantized.resize(hw.size());
    for (size_t i = 0; i < hw.size(); i++) r.dequantized[i] = __half2float(hw[i]);
    return r;
}

std::vector<int32_t> test_sample(const std::vector<float> &logits, int R, int V, const std::vector<SampleRequest> &reqs) {
    DeviceBuffer<float> dl(logits.size());
    DeviceBuffer<DeviceSampleParams> dp(static_cast<size_t>(R));
    DeviceBuffer<int32_t> dout(static_cast<size_t>(R));
    std::vector<DeviceSampleParams> hp;
    for (const auto &r : reqs)
        hp.push_back({r.params.temperature, r.params.top_p, r.params.min_p, r.params.top_k, r.params.seed, r.counter});
    CUDA_CHECK(cudaMemcpy(dl.get(), logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp.get(), hp.data(), hp.size() * sizeof(DeviceSampleParams), cudaMemcpyHostToDevice));
    sample_tokens(dl.get(), R, V, dp.get(), dout.get(), nullptr);
    std::vector<int32_t> out(static_cast<size_t>(R));
    CUDA_CHECK(cudaMemcpy(out.data(), dout.get(), out.size() * 4, cudaMemcpyDeviceToHost));
    return out;
}

}  // namespace cuda

int64_t cuda_free_bytes() {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) return 0;
    return static_cast<int64_t>(free);
}

bool cuda_device_present() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<Backend> create_cuda_backend(const std::string &dir, const BackendOptions &opt) {
    return std::make_unique<cuda::CudaBackend>(dir, opt);
}

}  // namespace ember
