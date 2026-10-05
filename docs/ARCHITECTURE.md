# Ember architecture

Ember turns a Hugging Face model directory into a server that answers many
requests at once, as fast as the GPU allows. This document follows a request
through the system and explains each design decision. Kernel-level details are
in [KERNELS.md](KERNELS.md); measurements in [BENCHMARKS.md](BENCHMARKS.md).

```
 clients ──HTTP──▶ server/      one thread per connection; OpenAI + Anthropic APIs, SSE
                     │ submit()          ▲ tokens (RequestStream)
                     ▼                   │
                   engine/      one thread: scheduler + continuous batching (+ speculative decoding)
                     │ StepBatch         ▲ sampled tokens
                     ▼                   │
                   backend/     Backend interface: CPU reference | CUDA
                     │                   
                   kvcache/     paged KV memory, prefix cache          model/  config, safetensors,
                                                                               tokenizer, chat template
```

## 1. From text to tokens (`src/model`)

- **Tokenizer.** A byte-level BPE compatible with Hugging Face's `tokenizer.json`:
  special tokens are split out first, the rest is NFC-normalized, cut into
  pieces by the Qwen pre-tokenizer regex, and each piece is merged with the BPE
  rank table. The regex is hand-written (no regex engine): it is about 20x
  faster and dependency-free, and the tests check it against the official
  tokenizer on hard inputs (Romanian diacritics, emoji, CJK, code, invalid
  UTF-8). Unicode tables (letters, numbers, white space, NFC data) are generated
  from Python's `unicodedata` by `scripts/gen_unicode_tables.py`.
- **Chat template.** Qwen3's Jinja template, reproduced in C++ (including how
  `<think>` blocks of earlier turns are dropped) and checked against
  `transformers.apply_chat_template`.
- **Weights.** `.safetensors` files are memory-mapped; tensors are views into
  the mapping. The CPU backend computes directly from them; the CUDA backend
  converts and uploads them once.

## 2. The engine (`src/engine`)

The engine owns the requests. A dedicated thread runs `step()` in a loop:

1. **Schedule.** Build one *ragged batch*: every sequence contributes a run of
   new tokens. Running sequences that are generating contribute one token each
   (decode first, so generation never stalls behind a long prompt); sequences
   still reading their prompt contribute a chunk (at most 512 tokens per
   sequence per step: *chunked prefill*); then new requests are admitted while
   the token budget (2048) and the KV cache allow.
2. **Forward.** The backend runs the model once over the batch.
3. **Sample.** One token per sequence that reached the end of its known tokens.
4. **Stream.** New tokens go to each request's `RequestStream`; finished
   requests release their memory.

A prompt chunk, a decode token and a speculative check are all the same
operation - "append these tokens to this sequence's cache and give me logits
for some of them" - which is what lets one step mix them freely.

**Preemption.** When the KV cache is full, the most recently started sequence
is preempted: its blocks are released and it goes back to the front of the
queue, to be recomputed later. With the prefix cache, its full blocks usually
survive, so the recomputation is mostly a cache hit.

**Cancellation.** A client that disconnects cancels its request; the engine
drops it at the next step and frees its blocks.

## 3. Paged KV cache and prefix caching (`src/kvcache`)

Keys and values are stored in blocks of 16 tokens (all layers). Each sequence
has a block table mapping its logical blocks to physical ones, like virtual
memory pages, so memory is allocated as sequences grow and nothing is reserved
for a maximum length.

A block completely filled is **sealed**: registered under a hash of its tokens
chained with the hash of the block before it (a key for the whole prefix). A
new request first looks up its prompt block by block; every hit is shared
(reference counted) instead of recomputed. When the last user of a sealed block
lets go, the block is not wiped: it waits in an LRU list, still findable, until
memory is needed. So:

- a system prompt shared by many requests is computed once;
- in a conversation, each turn reuses the previous turns (generated text is
  sealed too);
- a preempted sequence usually finds its own blocks again.

Hash collisions are ruled out by comparing the stored tokens. `engine.check()`
audits the accounting (every block exactly once free, cached or held, and held
as many times as referenced); the tests run it after every step.

## 4. The backends (`src/backend`)

`Backend` is an interface with two implementations:

- **CPU** - a direct, readable f32 implementation (AVX2 kernels for the matrix
  products, chosen at run time). It is the reference the CUDA backend is tested
  against, and it lets the engine and server run without a GPU (CI).
- **CUDA** - the fast one. Weights in f16, int8 or int4; KV cache in f16;
  the residual stream in f32 (Qwen3's activations reach ~1500, where f16 has a
  resolution of 1).

Weight layout choices made at load time keep the forward pass short:
q/k/v are one matrix (one GEMM instead of three), gate and up are interleaved
row by row so the GEMM epilogue computes `silu(gate) * up` in registers, and the
LM head is shared with the embedding when tied.

A layer is: RMSNorm → QKV GEMM → QK-norm + RoPE + cache write (one fused
kernel) → attention → output GEMM (added into the residual stream) → RMSNorm →
gate/up GEMM with fused SiLU → down GEMM (added into the residual).

The step's integers (tokens, positions, block tables, ...) are packed into one
pinned buffer and uploaded with a single copy.

### CUDA Graphs

A pure decode step (one token per sequence, up to 64 sequences) replays a
captured CUDA graph: about 300 kernel launches become one. Graphs freeze kernel
arguments, so each graph has a fixed layout for a bucket of batch sizes and
block-table widths; the batch is padded to the bucket, and padding tokens write
into a spare KV block that the engine never hands out.

### Batch invariance

Decode attention splits long contexts into fixed chunks of 512 keys
(`[0, 512), [512, 1024), ...`), merged in a fixed order, and the decode GEMMs
compute every token's column independently. A token is therefore computed with
the same arithmetic whatever else is in the batch (up to 64 decode tokens per
step): a request produces the same text alone or among others. This is what
makes speculative decoding reproduce plain decoding exactly on the GPU.

## 5. Sampling

Sampling uses the **Gumbel-max trick**: the next token is
`argmax_i(x_i / T + g_i)` over the tokens kept by top-k / top-p / min-p, where
`g_i` is Gumbel noise computed from `(seed, position, token id)` by a hash. This
is an exact sample from the softmax, needs no sort of the 150k-entry
vocabulary, parallelizes over it, and makes a sequence's random choices
independent of the batch. Top-k and top-p become logit thresholds, found by
radix selection over the float bits (4 passes) instead of sorting. Host and GPU
draw the same noise.

## 6. Quantization

Weight-only: int8 with one scale per row, int4 with a scale and a minimum per
group of 128 weights. The int4 quantizer searches 25 clippings of each group's
range and refits (scale, min) by least squares, keeping the lowest error.
Measured on Qwen3-1.7B, pure int4 costs about +50% perplexity, almost all of it
from the MLP down projection and q/k/v; Ember's `-q int4` therefore keeps those
(and the output layer) in int8 - the same idea as llama.cpp's Q4_K_M - for
about 0-5% perplexity at 2.2x the f16 decode speed. `--quant-mix` overrides the
recipe per matrix kind.

## 7. Speculative decoding

With `--draft`, a small model (e.g. Qwen3-0.6B) proposes k tokens per sequence;
the target checks all k + 1 positions in one forward pass. Both models sample
with the same Gumbel noise per position, so the target's sample at each
position is exactly what it would have drawn without speculation: draft tokens
are accepted while they equal the target's samples, and at the first
difference the target's own token is used. The output is identical to plain
decoding with the same seed (tested); only the number of target passes changes.
The draft keeps its own paged cache and catches up lazily (in budget-sized
chunks) before proposing.

This coupling trades some acceptance rate (compared with the classic rejection
scheme, which preserves only the distribution) for exact reproducibility. It
pays off most with greedy decoding and a slow (f16) target.

## 8. The server (`src/server`)

A small HTTP/1.1 server on raw sockets: one thread per connection, keep-alive,
chunked transfer encoding for Server-Sent Events. Handlers parse OpenAI or
Anthropic requests, render the chat template, submit to the engine and stream
the tokens back:

- Qwen3's `<think>` ... `</think>` are single tokens, so reasoning is split from
  the answer by token id (OpenAI: `reasoning_content`; Anthropic: a `thinking`
  block);
- text is streamed only in whole UTF-8 characters;
- stop strings are matched on the decoded text, holding back any tail that
  could still become one;
- a disconnected client cancels its request.

`/api/events` streams the engine's state to the dashboard (a single HTML file
compiled into the binary).
