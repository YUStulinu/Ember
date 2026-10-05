// The engine and the block manager, driven by a fake backend whose "model"
// reads its context back from the paged cache: any scheduling or block
// accounting mistake changes the generated tokens.
#include <map>
#include <random>
#include <set>

#include "engine/engine.hpp"
#include "kvcache/block_manager.hpp"
#include "test.hpp"

using namespace ember;

namespace {

constexpr int kVocab = 1000;
constexpr int kEos = 1;

// Next token after a context: a hash of every token, never EOS unless asked.
int32_t next_token(const std::vector<int32_t> &ctx, bool allow_eos, int disagree = 0) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : ctx) {
        h ^= static_cast<uint64_t>(t);
        h *= 1099511628211ull;
    }
    int32_t tok = static_cast<int32_t>(2 + (h >> 17) % (kVocab - 2));
    if (allow_eos && (h >> 7) % 23 == 0) tok = kEos;
    if (disagree && (h >> 40) % disagree == 0) tok = static_cast<int32_t>(2 + (tok + 7) % (kVocab - 2));  // a wrong guess
    return tok;
}

class FakeBackend final : public Backend {
public:
    FakeBackend(int blocks, int max_batch_tokens, int max_seqs, bool eos, int disagree = 0, int logit_rows = 0)
        : eos_(eos), disagree_(disagree) {
        cfg_.arch = "qwen3";
        cfg_.vocab_size = kVocab;
        cfg_.hidden = 8;
        cfg_.n_layers = 1;
        cfg_.n_heads = cfg_.n_kv_heads = 1;
        cfg_.head_dim = 8;
        cfg_.max_position = 100000;
        cfg_.eos_ids = {kEos};
        opt_.max_batch_tokens = max_batch_tokens;
        opt_.max_seqs = max_seqs;
        opt_.max_logit_rows = logit_rows ? logit_rows : max_seqs;
        cache_.assign(static_cast<size_t>(blocks) * kBlockSize, -1);
        blocks_ = blocks;
    }
    std::string name() const override { return "fake"; }
    const ModelConfig &config() const override { return cfg_; }
    const BackendOptions &options() const override { return opt_; }
    int num_kv_blocks() const override { return blocks_; }
    MemoryInfo memory() const override { return {}; }

    void forward(const StepBatch &b) override {
        CHECK(b.num_tokens() <= opt_.max_batch_tokens && b.num_seqs() <= opt_.max_seqs);
        CHECK(static_cast<int>(b.logit_rows.size()) <= opt_.max_logit_rows);
        steps++;
        max_seen_tokens = std::max(max_seen_tokens, b.num_tokens());
        // Write every new token into its slot first (as the real backends do).
        for (int s = 0; s < b.num_seqs(); s++)
            for (int t = b.query_start[s]; t < b.query_start[s + 1]; t++)
                cache_[slot(b, s, b.positions[static_cast<size_t>(t)])] = b.tokens[static_cast<size_t>(t)];
        next_.clear();
        for (int32_t row : b.logit_rows) {
            int s = 0;
            while (b.query_start[s + 1] <= row) s++;
            std::vector<int32_t> ctx;
            for (int p = 0; p <= b.positions[static_cast<size_t>(row)]; p++) ctx.push_back(cache_[slot(b, s, p)]);
            next_.push_back(next_token(ctx, eos_, disagree_));
        }
    }
    void sample(std::span<const SampleRequest> reqs, std::span<int32_t> out) override {
        CHECK(reqs.size() == next_.size());
        for (size_t i = 0; i < out.size(); i++) out[i] = next_[i];
    }
    void token_probs(std::span<const SampleRequest>, std::span<const int32_t>, std::span<const int32_t>, std::span<float>) override {}
    void logits(int, std::span<float>) override {}
    void set_capture_hidden(bool) override {}
    std::vector<float> captured_hidden() override { return {}; }

    int steps = 0, max_seen_tokens = 0;

private:
    size_t slot(const StepBatch &b, int s, int pos) const {
        int block = b.block_tables[static_cast<size_t>(s) * b.max_blocks + pos / kBlockSize];
        CHECK(block >= 0 && block < blocks_);
        return static_cast<size_t>(block) * kBlockSize + pos % kBlockSize;
    }
    ModelConfig cfg_;
    BackendOptions opt_;
    std::vector<int32_t> cache_;
    std::vector<int32_t> next_;
    int blocks_;
    bool eos_;
    int disagree_;
};

// What a request must produce, computed directly.
std::vector<int32_t> expected(std::vector<int32_t> ctx, int max_tokens, bool eos) {
    std::vector<int32_t> out;
    for (int i = 0; i < max_tokens; i++) {
        int32_t t = next_token(ctx, eos);
        if (t == kEos) break;
        out.push_back(t);
        ctx.push_back(t);
    }
    return out;
}

std::vector<int32_t> random_tokens(std::mt19937 &rng, int n) {
    std::uniform_int_distribution<int32_t> d(2, kVocab - 1);
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (auto &t : v) t = d(rng);
    return v;
}

struct Running {
    std::shared_ptr<RequestStream> stream;
    std::vector<int32_t> want, got;
    bool done = false;
};

// Steps the engine until every request finishes, auditing the blocks each step.
void drain(Engine &engine, std::vector<Running> &reqs) {
    for (int guard = 0; guard < 200000; guard++) {
        bool all = true;
        for (auto &r : reqs) {
            Delta d;
            while (r.stream->next(d, std::chrono::milliseconds(0))) {
                r.got.insert(r.got.end(), d.tokens.begin(), d.tokens.end());
                if (d.finished) r.done = true;
            }
            all = all && r.done;
        }
        if (all) return;
        engine.step();
        std::string problem = engine.check();
        CHECK_MSG(problem.empty(), "block accounting: {}", problem);
    }
    CHECK_MSG(false, "the engine did not finish");
}

}  // namespace

TEST("block manager: allocation, release, prefix matching, eviction") {
    BlockManager bm(8, true);
    std::vector<int32_t> toks(40);
    for (int i = 0; i < 40; i++) toks[static_cast<size_t>(i)] = 100 + i;
    std::vector<int> mine;
    uint64_t chain = 0;
    for (int b = 0; b < 2; b++) {
        mine.push_back(bm.allocate());
        chain = bm.seal(mine.back(), chain, std::span<const int32_t>(toks).subspan(static_cast<size_t>(b) * kBlockSize, kBlockSize));
    }
    CHECK_EQ(bm.available(), 6);
    auto hit = bm.match_prefix(toks);  // 40 tokens: 2 full cached blocks
    CHECK_EQ(hit.size(), 2u);
    CHECK(hit[0] == mine[0] && hit[1] == mine[1] && bm.refcount(mine[0]) == 2);
    bm.release_all(hit);
    bm.release_all(mine);
    CHECK_EQ(bm.available(), 8);       // free again...
    CHECK_EQ(bm.cached(), 2);          // ...but still findable
    auto again = bm.match_prefix(toks);
    CHECK_EQ(again.size(), 2u);
    bm.release_all(again);
    std::vector<int32_t> other(toks);
    other[3] = 7;                       // differs in the first block: no hit
    CHECK(bm.match_prefix(other).empty());
    std::vector<int> all;               // taking every block evicts the cached ones
    for (int i = 0; i < 8; i++) all.push_back(bm.allocate());
    CHECK(bm.allocate() == -1);
    CHECK_EQ(bm.cached(), 0);
    CHECK(bm.check().empty());
    bm.release_all(all);
    CHECK(bm.check().empty() && bm.available() == 8);
}

TEST("engine: one request matches the reference") {
    FakeBackend be(64, 256, 8, false);
    Engine engine(be, EngineOptions{});
    std::mt19937 rng(1);
    Request r;
    r.prompt = random_tokens(rng, 37);
    r.max_tokens = 50;
    auto got = engine.generate(r);
    CHECK(got == expected(r.prompt, 50, false));
}

TEST("engine: many concurrent requests, tiny cache, preemption, chunked prefill") {
    // 40 blocks = 640 tokens of cache for 60 requests of up to ~300 tokens each.
    for (int round = 0; round < 3; round++) {
        FakeBackend be(40, 96, 12, true);
        EngineOptions eo;
        eo.max_prefill_chunk = 40;
        Engine engine(be, eo);
        std::mt19937 rng(static_cast<uint32_t>(test::seed() + round));
        std::vector<Running> reqs;
        for (int i = 0; i < 60; i++) {
            Request r;
            r.prompt = random_tokens(rng, 1 + static_cast<int>(rng() % 200));
            r.max_tokens = 1 + static_cast<int>(rng() % 100);
            Running run;
            run.want = expected(r.prompt, r.max_tokens, true);
            run.stream = engine.submit(r);
            reqs.push_back(std::move(run));
        }
        drain(engine, reqs);
        for (size_t i = 0; i < reqs.size(); i++) CHECK_MSG(reqs[i].got == reqs[i].want, "request {} differs", i);
        EngineStats st = engine.stats();
        CHECK(st.preemptions > 0);           // the cache really was too small
        CHECK_EQ(st.kv_used, 0);             // everything released
        CHECK(be.max_seen_tokens <= 96);
    }
}

TEST("engine: shared prefixes are computed once") {
    FakeBackend be(200, 512, 8, false);
    Engine engine(be, EngineOptions{});
    std::mt19937 rng(5);
    auto system = random_tokens(rng, 300);  // a long shared "system prompt"
    std::vector<Running> reqs;
    for (int i = 0; i < 6; i++) {
        Request r;
        r.prompt = system;
        auto tail = random_tokens(rng, 5 + i);
        r.prompt.insert(r.prompt.end(), tail.begin(), tail.end());
        r.max_tokens = 20;
        Running run;
        run.want = expected(r.prompt, 20, false);
        run.stream = engine.submit(r);
        reqs.push_back(std::move(run));
        drain(engine, reqs);  // one after another: each finds the previous one's blocks
    }
    for (auto &r : reqs) CHECK(r.got == r.want);
    EngineStats st = engine.stats();
    CHECK_MSG(st.cached_tokens >= 5u * 288u, "only {} cached tokens", st.cached_tokens);  // 18 full blocks each after the first

    // A conversation: the next turn's prompt extends the previous prompt + answer.
    Request turn;
    turn.prompt = reqs[0].got;
    turn.prompt.insert(turn.prompt.begin(), system.begin(), system.end());
    auto before = engine.stats().cached_tokens;
    turn.prompt.push_back(7);
    std::vector<Running> one(1);
    one[0].want = expected(turn.prompt, 10, false);
    turn.max_tokens = 10;
    one[0].stream = engine.submit(turn);
    drain(engine, one);
    CHECK(one[0].got == one[0].want);
    CHECK(engine.stats().cached_tokens > before);
}

TEST("engine: cancellation frees the request's blocks") {
    FakeBackend be(64, 128, 8, false);
    Engine engine(be, EngineOptions{});
    std::mt19937 rng(9);
    Request r;
    r.prompt = random_tokens(rng, 50);
    r.max_tokens = 1000;
    auto stream = engine.submit(r);
    for (int i = 0; i < 10; i++) engine.step();
    CHECK(engine.stats().kv_used > 0);
    stream->cancel();
    engine.step();
    Delta d;
    bool finished = false;
    while (stream->next(d, std::chrono::milliseconds(0)))
        if (d.finished) finished = d.reason == FinishReason::cancelled;
    CHECK(finished);
    CHECK_EQ(engine.stats().kv_used, 0);
    CHECK(engine.check().empty());
}

TEST("engine: background loop serves concurrent submitters") {
    FakeBackend be(128, 256, 16, true);
    Engine engine(be, EngineOptions{});
    engine.start();
    std::mt19937 rng(11);
    std::vector<Running> reqs;
    for (int i = 0; i < 24; i++) {
        Request r;
        r.prompt = random_tokens(rng, 10 + i);
        r.max_tokens = 30;
        Running run;
        run.want = expected(r.prompt, 30, true);
        run.stream = engine.submit(r);
        reqs.push_back(std::move(run));
    }
    for (auto &r : reqs) {
        Delta d;
        while (!r.done) {
            if (!r.stream->next(d, std::chrono::milliseconds(2000))) break;
            r.got.insert(r.got.end(), d.tokens.begin(), d.tokens.end());
            r.done = d.finished;
        }
        CHECK(r.done && r.got == r.want);
    }
    engine.stop();
}

TEST("engine: speculative decoding gives exactly the target's output") {
    for (int disagree : {1000000, 3, 2}) {  // draft almost always right, often wrong, mostly wrong
        FakeBackend target(256, 256, 16, true, 0, 16 * 5);
        FakeBackend draft(256, 256, 16, true, disagree);
        EngineOptions eo;
        eo.spec_tokens = 4;
        Engine engine(target, eo, &draft);
        std::mt19937 rng(static_cast<uint32_t>(test::seed() + disagree));
        std::vector<Running> reqs;
        for (int i = 0; i < 12; i++) {
            Request r;
            r.prompt = random_tokens(rng, 5 + static_cast<int>(rng() % 60));
            r.max_tokens = 10 + static_cast<int>(rng() % 70);
            Running run;
            run.want = expected(r.prompt, r.max_tokens, true);
            run.stream = engine.submit(r);
            reqs.push_back(std::move(run));
        }
        drain(engine, reqs);
        for (size_t i = 0; i < reqs.size(); i++) CHECK_MSG(reqs[i].got == reqs[i].want, "disagree {}: request {} differs", disagree, i);
        EngineStats st = engine.stats();
        CHECK(st.spec_steps > 0);
        const double rate = static_cast<double>(st.spec_accepted) / std::max<uint64_t>(1, st.spec_proposed);
        if (test::verbose()) std::printf("[1/%d wrong: acceptance %.2f] ", disagree, rate);
        if (disagree > 1000) CHECK(rate > 0.9);
        CHECK_EQ(engine.stats().kv_used, 0);
    }
}
