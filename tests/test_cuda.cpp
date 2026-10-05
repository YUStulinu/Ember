// CUDA kernels against straightforward host references, then the whole CUDA
// model against PyTorch.
#include <algorithm>
#include <cmath>
#include <random>

#include "backend/sampling_host.hpp"
#include "common/fp16.hpp"
#include "test.hpp"

#ifdef EMBER_WITH_CUDA
#include "backend/cuda/cuda_testing.hpp"
#endif

using namespace ember;

#ifdef EMBER_WITH_CUDA

namespace {

std::vector<uint16_t> random_f16(size_t n, std::mt19937 &rng, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = float_to_fp16(d(rng));
    return v;
}

// epilogue: 0 store_f16, 1 store_f32, 2 add_f32, 3 silu_mul (matches cuda::Epilogue)
std::vector<float> reference(const std::vector<uint16_t> &A, const std::vector<uint16_t> &W, int M, int N, int K, int ep) {
    std::vector<float> full(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += static_cast<double>(fp16_to_float(A[static_cast<size_t>(m) * K + k])) * fp16_to_float(W[static_cast<size_t>(n) * K + k]);
            full[static_cast<size_t>(m) * N + n] = static_cast<float>(s);
        }
    if (ep == 2)
        for (auto &x : full) x += 1.0f;
    if (ep != 3) return full;
    std::vector<float> out(static_cast<size_t>(M) * (N / 2));
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N / 2; j++) {
            float g = full[static_cast<size_t>(m) * N + 2 * j], u = full[static_cast<size_t>(m) * N + 2 * j + 1];
            out[static_cast<size_t>(m) * (N / 2) + j] = g / (1.0f + std::exp(-g)) * u;
        }
    return out;
}

double max_rel_error(const std::vector<float> &got, const std::vector<float> &want) {
    double worst = 0, scale = 1e-3;
    for (float w : want) scale = std::max(scale, static_cast<double>(std::fabs(w)));
    for (size_t i = 0; i < got.size(); i++) worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]) / scale);
    return worst;
}

}  // namespace

TEST("cuda gemm: every kernel and epilogue matches the reference") {
    test::require_cuda();
    std::mt19937 rng(static_cast<uint32_t>(test::seed()));
    struct Shape {
        int M, N, K;
    };
    // Odd M and N exercise the edges; K covers several k-tiles.
    const Shape shapes[] = {{1, 256, 512}, {3, 130, 264}, {8, 512, 1024}, {13, 200, 136}, {31, 384, 512},
                            {64, 256, 1024}, {100, 300, 520}, {257, 512, 768}};
    const int kernels[] = {1, 2, 3, 4, 5, 6};  // gemv, tc16, tc32, tc64, tc128, gemv_tc
    for (const Shape &s : shapes) {
        auto A = random_f16(static_cast<size_t>(s.M) * s.K, rng, 1.0f);
        auto W = random_f16(static_cast<size_t>(s.N) * s.K, rng, 0.05f);
        for (int ep = 0; ep < 4; ep++) {
            if (ep == 3 && s.N % 2) continue;
            auto want = reference(A, W, s.M, s.N, s.K, ep);
            for (int kernel : kernels) {
                if (kernel == 1 && s.M > 8) continue;
                if (kernel == 6 && (s.M > 64 || s.K % 32)) continue;
                for (bool split : {false, true}) {
                    if (split && (kernel == 1 || kernel == 6)) continue;
                    auto got = cuda::test_gemm(A, W, s.M, s.N, s.K, ep, kernel, split);
                    double err = max_rel_error(got, want);
                    CHECK_MSG(err < 2e-3, "M={} N={} K={} epilogue {} kernel {} split {}: error {:.2e}", s.M, s.N, s.K,
                              ep, kernel, split, err);
                }
            }
        }
    }
}

TEST("cuda quantization: kernels match the dequantized weights; error is bounded") {
    test::require_cuda();
    std::mt19937 rng(static_cast<uint32_t>(test::seed()));
    for (int qt : {1, 2}) {  // int8, int4
        for (int M : {1, 5, 16, 33, 64}) {
            const int N = 96, K = 512;
            auto A = random_f16(static_cast<size_t>(M) * K, rng, 1.0f);
            auto W = random_f16(static_cast<size_t>(N) * K, rng, 0.05f);
            for (int ep = 0; ep < 4; ep++) {
                auto r = cuda::test_quant_gemm(A, W, M, N, K, qt, ep);
                double err = max_rel_error(r.quant, r.reference);
                CHECK_MSG(err < 3e-3, "qt {} M={} epilogue {}: kernel differs from dequantized GEMM by {:.2e}", qt, M, ep, err);
                if (ep != 0 || M != 1) continue;
                // Reconstruction: int8 within half a step of each row; int4 (with its clipping
                // search) never worse, per group, than plain min/max rounding.
                const int group = qt == 2 ? 128 : K;
                for (int n = 0; n < N; n++)
                    for (int g0 = 0; g0 < K; g0 += group) {
                        float lo = 1e9f, hi = -1e9f, amax = 0;
                        for (int k = g0; k < g0 + group; k++) {
                            float w = fp16_to_float(W[static_cast<size_t>(n) * K + k]);
                            lo = std::min(lo, w);
                            hi = std::max(hi, w);
                            amax = std::max(amax, std::fabs(w));
                        }
                        double sse = 0, sse_rtn = 0;
                        for (int k = g0; k < g0 + group; k++) {
                            float w = fp16_to_float(W[static_cast<size_t>(n) * K + k]);
                            float d = r.dequantized[static_cast<size_t>(n) * K + k];
                            sse += (w - d) * (w - d);
                            if (qt == 1) {
                                const float step = amax / 127;
                                CHECK_MSG(std::fabs(w - d) <= 0.5f * step * 1.02f + 1e-4f, "int8 row {} k {}: {} -> {}", n, k, w, d);
                            } else {
                                const float step = (hi - lo) / 15;
                                float q = std::round((w - lo) / step);
                                float rt = q * step + lo;
                                sse_rtn += (w - rt) * (w - rt);
                            }
                        }
                        if (qt == 2) CHECK_MSG(sse <= sse_rtn * 1.02 + 1e-9, "int4 row {} group {}: error {} > rounding {}", n, g0, sse, sse_rtn);
                    }
            }
        }
    }
}

TEST("cuda sampling: greedy, filters and the sampled distribution") {
    test::require_cuda();
    const int V = 5000;
    std::mt19937 rng(7);
    std::normal_distribution<float> d(0.0f, 3.0f);
    std::vector<float> logits(static_cast<size_t>(V));
    for (auto &x : logits) x = d(rng);

    // Greedy, and top_k = 1, both pick the argmax.
    int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<SampleRequest> reqs(2);
    reqs[0].params.temperature = 0;
    reqs[1].params.temperature = 1.0f;
    reqs[1].params.top_k = 1;
    std::vector<float> two(logits);
    two.insert(two.end(), logits.begin(), logits.end());
    auto got = cuda::test_sample(two, 2, V, reqs);
    CHECK_EQ(got[0], best);
    CHECK_EQ(got[1], best);

    // Many draws with top-k/top-p: every token must be in the host's kept set,
    // and the frequencies must follow the host distribution.
    SamplingParams p;
    p.temperature = 0.8f;
    p.top_k = 40;
    p.top_p = 0.9f;
    auto dist = sampling_distribution(logits.data(), V, p);
    const int R = 2000;
    std::vector<float> many;
    std::vector<SampleRequest> rs;
    for (int r = 0; r < R; r++) {
        many.insert(many.end(), logits.begin(), logits.end());
        SampleRequest q;
        q.params = p;
        q.params.seed = 99;
        q.counter = static_cast<uint64_t>(r);
        rs.push_back(q);
    }
    auto toks = cuda::test_sample(many, R, V, rs);
    std::vector<int> hist(static_cast<size_t>(V));
    for (int t : toks) hist[static_cast<size_t>(t)]++;
    double kept_mass = 0;
    for (const auto &[tok, prob] : dist) kept_mass += hist[static_cast<size_t>(tok)];
    CHECK_EQ(kept_mass, static_cast<double>(R));  // nothing outside the kept set
    // The most likely token's frequency is within a few standard deviations.
    double p0 = dist[0].second, f0 = static_cast<double>(hist[static_cast<size_t>(dist[0].first)]) / R;
    CHECK_MSG(std::fabs(f0 - p0) < 4 * std::sqrt(p0 * (1 - p0) / R) + 0.01, "top token: frequency {:.3f}, probability {:.3f}", f0, p0);
    // Same seed and counter, same token: the GPU and host samplers agree.
    for (int r = 0; r < 50; r++) CHECK_EQ(toks[static_cast<size_t>(r)], sample_host(logits.data(), V, rs[static_cast<size_t>(r)]));
}

#endif  // EMBER_WITH_CUDA
