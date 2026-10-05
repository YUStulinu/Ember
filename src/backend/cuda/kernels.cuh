// Kernel launchers of the CUDA backend. Each one enqueues work on `stream`
// and returns immediately.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ember::cuda {

// ---- weights ----------------------------------------------------------------

enum class SrcType { f32, f16, bf16 };

// Converts `rows` rows of `cols` elements to f16, writing source row r to
// destination row dst_offset + r * dst_stride. Values beyond the f16 range are
// clamped and counted in *overflow.
void convert_rows_to_f16(const void *src, SrcType type, int64_t rows, int64_t cols, half *dst, int64_t dst_offset,
                         int64_t dst_stride, unsigned long long *overflow, cudaStream_t stream);
void convert_to_f32(const void *src, SrcType type, int64_t n, float *dst, cudaStream_t stream);

// ---- matrix multiplication: C[M, N] = A[M, K] * W[N, K]^T ----------------------

enum class Epilogue {
    store_f16,  // C is half [M, ldc]
    store_f32,  // C is float [M, ldc]
    add_f32,    // C is float [M, ldc], C += result (the residual stream)
    silu_mul,   // W rows interleave gate/up; C is half [M, ldc] with C[m, j] = silu(gate_j) * up_j
};

enum class GemmKernel { automatic, gemv, tc16, tc32, tc64, tc128, gemv_tc };

// Scratch for split-K partial sums (fp32). Without it, no split.
struct GemmWorkspace {
    float *partial = nullptr;
    size_t partial_elems = 0;
};

void gemm(const half *A, int lda, const half *W, int M, int N, int K, void *C, int ldc, Epilogue ep, cudaStream_t stream,
          GemmKernel kernel = GemmKernel::automatic, const GemmWorkspace *ws = nullptr);

// Quantized weights (weight-only): int8 with one fp32 scale per row, or int4
// (packed as described in k_quant.cu) with an f16 (scale, min) pair per group
// of 128 along K. `scales` points to whichever applies.
enum class QuantType { none, int8, int4 };
constexpr int kInt4Group = 128;

struct QuantScratch {
    half *dequant = nullptr;     // f16 [N, K] for the prefill path (M > 64)
    size_t dequant_elems = 0;
    float *xsum = nullptr;       // per-token group sums of the input (int4)
    size_t xsum_elems = 0;
    const GemmWorkspace *ws = nullptr;
};

void gemm_quant(const half *A, int lda, const void *Wq, const half *scales, QuantType qt, int M, int N, int K, void *C,
                int ldc, Epilogue ep, const QuantScratch &scratch, cudaStream_t stream);
// Quantizes f16 rows [N, K] into Wq / scales (round to nearest).
void quantize_weights(const half *W, int N, int K, QuantType qt, void *Wq, half *scales, cudaStream_t stream);
// Expands quantized weights back to f16.
void dequantize_weights(const void *Wq, const half *scales, QuantType qt, int N, int K, half *out, cudaStream_t stream);
void embed_tokens_quant(const int32_t *tokens, int T, const void *Wq, const half *scales, QuantType qt, int hidden, float *out,
                        cudaStream_t stream);

// ---- transformer pieces -------------------------------------------------------

void embed_tokens(const int32_t *tokens, int T, const half *table, int hidden, float *out, cudaStream_t stream);

// out[r] = rmsnorm(x[rows ? rows[r] : r]) * w, for r < R.  x f32, out f16.
void rmsnorm_rows(const float *x, const int32_t *rows, int R, int hidden, const float *w, float eps, half *out,
                  cudaStream_t stream);

struct QkvArgs {
    const half *qkv;          // [T, (n_heads + 2 * n_kv) * D]: q heads, then k heads, then v heads
    half *q_out;              // [T, n_heads * D]
    half *k_cache, *v_cache;  // this layer's cache: [blocks][n_kv][kBlockSize][D]
    const float *q_norm, *k_norm;  // [D] or null
    const float *bias;        // [(n_heads + 2 * n_kv) * D] or null (Qwen2 q/k/v biases)
    const float *inv_freq;    // [D / 2]
    const int32_t *positions, *token_seq, *block_tables;
    int max_blocks, T, n_heads, n_kv, head_dim;
    float eps;
};
// QK-norm (if enabled), rotary embedding, and the cache write of K and V.
void qk_norm_rope_store(const QkvArgs &a, cudaStream_t stream);

struct AttentionArgs {
    const int32_t *token_index;      // [T] which tokens to process (null = 0..T-1)
    const half *q;                   // [T, n_heads, D]
    const half *k_cache, *v_cache;   // this layer's cache
    const int32_t *positions, *token_seq, *block_tables;
    int max_blocks, T, n_heads, n_kv, head_dim;
    float scale;
    half *out;                       // [T, n_heads, D]
    // Split-K over the context ("flash decoding"): each (token, head) context is
    // cut into chunks of split_len keys processed in parallel, then combined.
    int num_splits;                  // 1 = no split
    int split_len;
    float *part_acc, *part_ml;       // scratch for splits: [T, n_heads, splits, D] and [T, n_heads, splits, 2]
};
void paged_attention(const AttentionArgs &a, cudaStream_t stream);

// Prefill attention with tensor cores (FlashAttention-2 style), for sequences
// contributing many tokens. Each block handles a tile of 64 consecutive query
// tokens of one sequence for one head, streaming its keys and values from the
// paged cache 32 at a time.
constexpr int kPrefillTile = 64;
struct PrefillArgs {
    const half *q;                   // [T, n_heads, D]
    const half *k_cache, *v_cache;
    const int32_t *tile_start;       // [num_tiles] first token of the tile
    const int32_t *tile_end;         // [num_tiles] one past its last token
    const int32_t *tile_seq;         // [num_tiles] sequence of the tile
    const int32_t *positions, *block_tables;
    int num_tiles, max_blocks, n_heads, n_kv, head_dim;
    float scale;
    half *out;                       // [T, n_heads, D]
};
void prefill_attention(const PrefillArgs &a, cudaStream_t stream);

// ---- sampling --------------------------------------------------------------------

struct DeviceSampleParams {
    float temperature, top_p, min_p;
    int top_k;
    uint64_t seed, counter;
};
// One token per row of logits [R, V].
void sample_tokens(const float *logits, int R, int V, const DeviceSampleParams *params, int32_t *out, cudaStream_t stream);
// Probability of tokens[i] in the sampling distribution of row rows[i].
void token_probabilities(const float *logits, int V, const DeviceSampleParams *params, const int32_t *rows,
                         const int32_t *tokens, int n, float *out, cudaStream_t stream);

int sm_count();

}  // namespace ember::cuda
