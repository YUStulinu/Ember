# Benchmarks

All numbers: Qwen3-1.7B on an **NVIDIA RTX 2060 laptop GPU** (6 GB, 30 SMs,
~264 GB/s theoretical memory bandwidth), Intel i7-10870H, Windows 11, CUDA
13.4, on AC power. A laptop GPU changes its clocks with temperature, so every
figure is a median of 3 runs and comparisons alternate the two engines with
cool-down pauses in between.

## Ember vs llama.cpp

Same machine, same model, comparable formats: Ember `f16` / `int8` / `int4`
against llama.cpp `F16` / `Q8_0` / `Q4_K_M` (build b11429, CUDA 13.4 binaries,
`-ngl 99`, flash attention on). Prefill is a 512-token prompt (`llama-bench
pp512`). Generation is tokens per second once the prompts are read - for
llama.cpp the `S_TG` column of `llama-batched-bench` with 64-token prompts and
128 generated tokens per sequence; for Ember the "generation" column of
`ember bench` with the same shape.

| Format (Ember / llama.cpp) | Metric | Ember | llama.cpp | Ratio |
|---|---|---:|---:|---:|
| f16 / F16 | prefill, 512 tokens (t/s) | 3,741 | 5,166 | **0.72x** |
| f16 / F16 | generation, 1 sequence (t/s) | 55 | 11 | **4.94x** |
| f16 / F16 | generation, 8 sequences (t/s) | 425 | 25 | **17.32x** |
| f16 / F16 | generation, 32 sequences (t/s) | 1,186 | 198 | **6.01x** |
| int8 / Q8_0 | prefill, 512 tokens (t/s) | 3,059 | 4,031 | **0.76x** |
| int8 / Q8_0 | generation, 1 sequence (t/s) | 104 | 85 | **1.22x** |
| int8 / Q8_0 | generation, 8 sequences (t/s) | 766 | 406 | **1.89x** |
| int8 / Q8_0 | generation, 32 sequences (t/s) | 2,067 | 1,031 | **2.01x** |
| int4 / Q4_K_M | prefill, 512 tokens (t/s) | 3,128 | 3,656 | **0.86x** |
| int4 / Q4_K_M | generation, 1 sequence (t/s) | 126 | 107 | **1.18x** |
| int4 / Q4_K_M | generation, 8 sequences (t/s) | 900 | 342 | **2.63x** |
| int4 / Q4_K_M | generation, 32 sequences (t/s) | 1,578 | 994 | **1.59x** |

Notes:

- Ember's `int4` keeps q/k/v, the down projection and the output layer in int8
  (1275 MiB of weights); Q4_K_M is a little smaller (1.03 GiB), so this row
  compares recipes of similar quality rather than identical sizes.
- llama.cpp's batched F16 decode measured abnormally slow on this GPU (11 t/s
  for one sequence, while `llama-bench tg128` gives 55 t/s for the same model);
  the f16 rows are therefore not a fair comparison and are shown for
  completeness. The int8 and int4 rows are the meaningful ones.

## Ember alone

`ember bench` reports, per format: prefill throughput, then for 1, 8 and 32
concurrent sequences the generation throughput, the end-to-end throughput
(prompts included) and the time to the first token.

| Format | Weights | Prefill (512) | 1 sequence | 8 sequences | 32 sequences |
|---|---:|---:|---:|---:|---:|
| f16 | 3282 MiB | ~4,400 t/s | ~66 t/s | ~480 t/s | ~1,230 t/s* |
| int8 | 1752 MiB | ~3,100-3,450 t/s | ~107-122 t/s | ~785 t/s | ~2,100 t/s |
| int4 (recipe) | 1275 MiB | ~3,100-3,600 t/s | ~129-156 t/s | ~810-960 t/s | ~1,500-1,700 t/s* |

\* end-to-end throughput (prompts included); the others are generation
throughput. Ranges are the spread between a cool and a warm GPU.

### Speculative decoding

Target Qwen3-1.7B f16, draft Qwen3-0.6B int4, one sequence, greedy:

| Draft tokens per step (k) | Accepted | Tokens per target pass | Speed |
|---:|---:|---:|---:|
| none | - | 1 | 64.0 t/s |
| 2 | 68% | 2.36 | 89.5 t/s |
| 4 | 52% | 3.08 | 93.5 t/s |
| 6 | 45% | 3.68 | 99.7 t/s (1.56x) |

With sampling (temperature 0.7) acceptance drops to 35-50%: the coupling that
makes the output identical to plain decoding accepts less than classic
rejection sampling. With an int4 target the 0.6B draft costs too much relative
to the target to help.

### Quality (perplexity)

`ember perplexity` over non-overlapping 1024-token windows; English text is the
Python tutorial, Romanian text a held-out Wikipedia sample
(`reference/make_eval_text.py`).

| Weights | English | Romanian |
|---|---:|---:|
| f16 | 13.20 | 13.03 |
| int8 | 13.09 | 12.94 |
| int4, Ember's recipe | 13.19 | 13.70 |
| int4, all but the output layer | 19.76 | 22.32 |

Per-matrix sensitivity, pure int4 except the named matrix kept in f16
(English): q/k/v 17.6, output projection 20.5, gate/up 20.3, **down 14.4** -
which is why the recipe keeps down and q/k/v in int8.

### GEMM kernels

`ember kernels` (effective weight bandwidth, GB/s; theoretical 264):

| Matrix | M = 1 | M = 8 | M = 32 |
|---|---:|---:|---:|
| qkv 4096 x 2048 | 222 | 226 | 188 |
| gate/up 12288 x 2048 | 249 | 248 | 234 |
| down 2048 x 6144 | 193 | 222 | 116 |
| lm head 151936 x 2048 | 250 | 229 | 211 |

## Reproducing

```bash
cmake -S . -B build && cmake --build build --config Release
bash models/download.sh
build/ember bench -q int4                        # Ember alone
build/ember kernels                              # GEMM bandwidth
python reference/make_eval_text.py && build/ember perplexity -q int4 eval/en.txt eval/ro.txt
```

The comparison: put llama.cpp's release in `bench/external/llama.cpp/` and the
GGUF files (`Qwen3-1.7B-F16.gguf` - converted from the BF16 one with
`llama-quantize ... F16` - `Qwen3-1.7B-Q8_0.gguf`, `Qwen3-1.7B-Q4_K_M.gguf`,
from `unsloth/Qwen3-1.7B-GGUF`) in `bench/external/gguf/`, then
`python bench/compare.py`, which writes `bench/results/compare.md`.
