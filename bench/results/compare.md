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
