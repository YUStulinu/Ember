# Ember's CUDA kernels

All kernels are in `src/backend/cuda/`. They target Turing (sm_75, e.g. an RTX
2060) and later: tensor cores are used through `mma.sync.m16n8k8` and
`ldmatrix`, the instructions Turing has (no `cp.async`, no bf16).

Measured on an RTX 2060 laptop GPU (6 GB, 30 SMs, ~264 GB/s theoretical
memory bandwidth) with `ember kernels`, which times each GEMM variant on the
model's real shapes.

## Why decode and prefill need different kernels

Generating one token per sequence multiplies each weight matrix by a handful
of vectors: the GPU's work is reading the weights, and arithmetic is nearly
free. A step's time is about *weight bytes / memory bandwidth* - for
Qwen3-1.7B in f16, 3.3 GB / 264 GB/s = 12.4 ms, i.e. at most ~80 tokens/s for
one sequence. Reading a prompt multiplies the same weights by hundreds of
vectors at once: then arithmetic dominates, and tensor cores matter.

## Matrix multiplication: `C[M, N] = A[M, K] · W[N, K]ᵀ` (k_gemm.cu)

### `gemv_tc` - decode-sized batches (M ≤ 64)

Built around the loads, not the arithmetic: no shared-memory staging and no
`__syncthreads` in the main loop.

The trick: a dot product does not care about the order of k, so A and W may use
any common permutation of k. Thread `(g = lane / 4, t = lane % 4)` loads 16
contiguous bytes of a weight row, `W[row][kb + 8t .. kb + 8t + 7]`, and uses
its half2 number `j` as the mma fragment for "virtual" k = 2t, 2t + 1 of step
`j`. Loading the activations with the same pattern lines both operands up, so
fully coalesced 16-byte loads feed `mma.sync` directly. The roles are swapped
relative to a normal GEMM (16 weight rows are the mma's M side, 8 tokens its N
side), and the four warps of a block split K and add their partial sums at the
end.

Weights are loaded with `ld.global.nc.L1::no_allocate`: each byte is used once
per step and should not evict activations from L1.

| Shape (Qwen3-1.7B) | M = 1 | M = 8 | M = 32 |
|---|---:|---:|---:|
| qkv (4096 x 2048) | 222 GB/s | 226 GB/s | 188 GB/s |
| gate/up (12288 x 2048) | 249 GB/s | 248 GB/s | 234 GB/s |
| down (2048 x 6144) | 193 GB/s | 222 GB/s | 116 GB/s |
| lm head (151936 x 2048) | 250 GB/s | 229 GB/s | 211 GB/s |

That is 85-95% of the theoretical bandwidth for one to sixteen sequences.
(`down` at M = 32 is limited by re-reading its large input across blocks.)

### `gemm_tc` - prefill (M > 64)

A classic tiled tensor-core GEMM: 128x128 output tiles, 8 warps of 64x32,
k-tiles of 32 staged in shared memory with a padded row stride (40 halves =
80 bytes, so the eight rows an `ldmatrix` reads land in distinct banks), double
buffered through registers (the next tile is fetched while the current one is
multiplied). Smaller tile shapes (16/32/64 rows) and split-K exist for other
shapes. About 15-16 TFLOP/s on the largest matrices of a 512-token chunk.

### Fused epilogues

Every GEMM ends in one of four epilogues, applied while the results are still
in registers:

- `store_f16` - q/k/v projections;
- `store_f32` - logits;
- `add_f32` - output and down projections, added straight into the f32
  residual stream (no separate add kernel, no extra pass over memory);
- `silu_mul` - the MLP: gate and up rows are interleaved at load time, so each
  thread holds `(gate_j, up_j)` side by side and writes `silu(gate_j) * up_j`.

## Quantized decode (k_quant.cu)

Same design as `gemv_tc`, with two twists:

- the tensor cores multiply the **integer codes**: -127..127 and 0..15 are
  exact in f16, so the per-row (int8) or per-group (int4) scale is applied to
  the accumulated sums, not to every weight;
- int8 → f16 and int4 → f16 use the "magic number" trick: put the code in the
  mantissa of 1024.0 (`0x6400`) with `prmt` / a mask, subtract 1024 (+128 for
  int8) - two weights per instruction.

int4 weights are packed so that `(word >> 4j) & 0x000F000F` is the pair
`(k = 2j, 2j + 1)`: one shift and one mask per pair. A thread reads exactly one
group (32 weights of a 128-weight group per thread, 4 threads per group), so
the group's scale and minimum apply to one mma accumulation:
`s · Σ q·x + m · Σ x`, with `Σ x` per group precomputed by a tiny kernel.

Prefill with quantized weights expands a layer's matrices to f16 into a scratch
buffer and uses the tensor-core GEMM.

## Attention (k_attention.cu)

### Prefill: FlashAttention-2 on tensor cores

A block = 4 warps = 64 query rows of one sequence, one head. Q stays in
registers as mma fragments. For each tile of 32 keys (two pages of the cache):

1. K and V tiles are loaded into shared memory (padded rows);
2. `S = Q Kᵀ` with `ldmatrix` + `mma` (16 x 32 per warp);
3. causal mask, then an online softmax in registers (row max and sum across the
   four lanes that share a row);
4. `O += P V`: P is reused straight from S's accumulators as A-fragments (the
   accumulator layout of m16n8 equals the A-fragment layout of m16n8k8), and V
   is read with `ldmatrix.trans` as the B operand.

The full score matrix is never materialized. Slots past the context are loaded
as zeros: never-written cache memory could hold NaN bit patterns, and 0 · NaN
is NaN.

### Decode: paged attention with fixed splits

One block per (query token, kv head, context split). The query heads that share
a kv head (GQA) are processed together, so each key and value is read once for
all of them. Each warp walks whole pages (16 keys, contiguous):

- scores: two lanes per key, each covering half of `head_dim`, one shuffle;
- values: each lane owns `head_dim / 32` output dimensions;
- softmax online, in base 2, with `scale · log2(e)` folded into q.

Long contexts are cut into fixed 512-key splits merged by a second kernel
("flash decoding"). Fixed - not sized to the batch - so that a token's result
does not depend on what else is in the batch (see ARCHITECTURE.md, batch
invariance).

## The rest (k_elementwise.cu, k_sampling.cu)

- **QK-norm + RoPE + cache write** in one kernel, one warp per head: a lane
  holds `head_dim / 32` elements; the rotary partner `d ± head_dim / 2` lives in
  lane `l ^ 16`, one shuffle away.
- **RMSNorm** in f32 from the residual stream, writing f16 for the next GEMM.
- **Sampling** (one block per row): Gumbel-max over the kept tokens; top-k and
  top-p thresholds by radix selection over the float bits (4 passes of 8 bits,
  histograms in shared memory) instead of sorting 152k values.

## Testing

Every kernel variant is tested against a straightforward host reference
(`tests/test_cuda.cpp`): all GEMM kernels x all epilogues x awkward shapes
(odd M and N, several k-tiles), with and without split-K; the quantized kernels
against the f16 GEMM on the dequantized weights; the sampler's empirical
distribution against the host's. The whole CUDA model is checked layer by layer
against PyTorch and against the CPU backend on long prompts in uneven chunks
(`tests/test_model.cpp`).
