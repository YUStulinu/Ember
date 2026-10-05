#include "engine/engine.hpp"

#include <algorithm>
#include <random>

#include "common/error.hpp"
#include "common/log.hpp"

namespace ember {

const char *finish_reason_name(FinishReason r) {
    switch (r) {
        case FinishReason::none: return "none";
        case FinishReason::stop: return "stop";
        case FinishReason::length: return "length";
        case FinishReason::cancelled: return "cancelled";
        case FinishReason::error: return "error";
    }
    return "?";
}

// ---- RequestStream -------------------------------------------------------------

void RequestStream::push(const Delta &d) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(d);
    }
    cv_.notify_all();
}

bool RequestStream::next(Delta &out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [&] { return !queue_.empty(); })) return false;
    out = std::move(queue_.front());
    queue_.pop_front();
    // Merge whatever else is already there, so a slow reader catches up in one go.
    while (!queue_.empty()) {
        Delta &d = queue_.front();
        out.tokens.insert(out.tokens.end(), d.tokens.begin(), d.tokens.end());
        out.finished = d.finished;
        out.reason = d.reason;
        out.generated = d.generated;
        out.cached_tokens = d.cached_tokens;
        if (d.first_token_ms > 0) out.first_token_ms = d.first_token_ms;
        queue_.pop_front();
    }
    return true;
}

// ---- Engine ------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

struct Engine::Sequence {
    uint64_t id;
    Request req;
    std::shared_ptr<RequestStream> stream;
    std::vector<int32_t> tokens;  // prompt + generated
    int prompt_len = 0;
    int computed = 0;             // tokens whose keys/values are in the cache
    std::vector<int> blocks;
    int sealed = 0;               // leading blocks registered in the prefix cache
    uint64_t chain = 0;           // hash of the last sealed block
    int generated = 0;
    int cached = 0;               // prompt tokens taken from the prefix cache (first admission)
    bool admitted_once = false;
    Clock::time_point arrival;
    double first_token_ms = 0;
    std::vector<int32_t> pending;  // tokens not yet delivered to the stream
    std::vector<int> draft_blocks; // the draft model's cache (speculative decoding)
    int draft_computed = 0;
};

Engine::Engine(Backend &backend, const EngineOptions &options, Backend *draft)
    : backend_(backend), opt_(options), blocks_(backend.num_kv_blocks(), options.prefix_caching), draft_(draft),
      start_time_(Clock::now()) {
    if (draft_) {
        EMBER_CHECK(draft_->config().vocab_size == backend_.config().vocab_size,
                    "the draft model's vocabulary ({}) differs from the target's ({})", draft_->config().vocab_size,
                    backend_.config().vocab_size);
        EMBER_CHECK(opt_.spec_tokens >= 1 && opt_.spec_tokens <= 16, "--spec-tokens must be in 1..16");
        draft_blocks_ = std::make_unique<BlockManager>(draft_->num_kv_blocks(), false);
    }
    const BackendOptions &bo = backend_.options();
    if (opt_.max_batch_tokens <= 0 || opt_.max_batch_tokens > bo.max_batch_tokens) opt_.max_batch_tokens = bo.max_batch_tokens;
    if (opt_.max_seqs <= 0 || opt_.max_seqs > bo.max_seqs) opt_.max_seqs = bo.max_seqs;
    int rows = bo.max_logit_rows > 0 ? bo.max_logit_rows : bo.max_seqs;
    opt_.max_seqs = std::min(opt_.max_seqs, rows);
    if (draft_) opt_.max_seqs = std::min(opt_.max_seqs, draft_->options().max_seqs);
    if (opt_.max_context <= 0 || opt_.max_context > backend_.config().max_position) opt_.max_context = backend_.config().max_position;
    opt_.max_context = std::min(opt_.max_context, backend_.num_kv_blocks() * kBlockSize);
    if (draft_) opt_.max_context = std::min(opt_.max_context, draft_->num_kv_blocks() * kBlockSize - opt_.spec_tokens - 1);
    opt_.max_prefill_chunk = std::max(1, std::min(opt_.max_prefill_chunk, opt_.max_batch_tokens));
    eos_ = backend_.config().eos_ids;
    stats_.kv_blocks = backend_.num_kv_blocks();
}

Engine::~Engine() { stop(); }

std::shared_ptr<RequestStream> Engine::submit(Request req) {
    EMBER_CHECK(!req.prompt.empty(), "the prompt is empty");
    EMBER_CHECK(static_cast<int>(req.prompt.size()) < opt_.max_context, "the prompt has {} tokens; the context limit is {}",
                req.prompt.size(), opt_.max_context);
    for (int32_t t : req.prompt)
        EMBER_CHECK(t >= 0 && t < backend_.config().vocab_size, "token id {} is out of range", t);
    req.max_tokens = std::max(1, req.max_tokens);
    if (req.sampling.seed == 0) {
        static std::atomic<uint64_t> counter{0x5EED};
        req.sampling.seed = (static_cast<uint64_t>(Clock::now().time_since_epoch().count()) * 0x9E3779B97F4A7C15ull) ^ counter++;
    }
    auto seq = std::make_unique<Sequence>();
    seq->req = std::move(req);
    seq->tokens = seq->req.prompt;
    seq->prompt_len = static_cast<int>(seq->tokens.size());
    seq->arrival = Clock::now();
    std::shared_ptr<RequestStream> stream;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        seq->id = next_id_++;
        stream = std::make_shared<RequestStream>(seq->id);
        seq->stream = stream;
        incoming_.push_back(std::move(seq));
    }
    wake_.notify_all();
    return stream;
}

void Engine::start() {
    if (started_) return;
    started_ = true;
    stop_ = false;
    thread_ = std::thread([this] { loop(); });
}

void Engine::stop() {
    if (!started_) return;
    stop_ = true;
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
    started_ = false;
}

void Engine::loop() {
    while (!stop_) {
        bool worked = false;
        try {
            worked = step();
        } catch (const std::exception &e) {
            // A failed step (e.g. a CUDA error) fails the requests in it instead of killing the server.
            log::error("engine step failed: {}", e.what());
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto &s : running_) {
                s->pending.clear();
                Delta d;
                d.finished = true;
                d.reason = FinishReason::error;
                d.generated = s->generated;
                s->stream->push(d);
                blocks_.release_all(s->blocks);
            }
            running_.clear();
        }
        if (!worked) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait_for(lock, std::chrono::milliseconds(50), [&] { return stop_ || !incoming_.empty(); });
        }
    }
}

void Engine::take_submissions() {
    for (auto &s : incoming_) waiting_.push_back(std::move(s));
    incoming_.clear();
}

bool Engine::ensure_blocks(Sequence &s, int upto) {
    const size_t need = static_cast<size_t>((upto + kBlockSize - 1) / kBlockSize);
    while (s.blocks.size() < need) {
        int b = blocks_.allocate();
        if (b < 0) return false;
        s.blocks.push_back(b);
    }
    return true;
}

void Engine::seal_full_blocks(Sequence &s) {
    if (!blocks_.prefix_caching()) return;
    while ((s.sealed + 1) * kBlockSize <= s.computed) {
        auto chunk = std::span<const int32_t>(s.tokens).subspan(static_cast<size_t>(s.sealed) * kBlockSize, kBlockSize);
        s.chain = blocks_.seal(s.blocks[static_cast<size_t>(s.sealed)], s.chain, chunk);
        s.sealed++;
    }
}

void Engine::preempt(Sequence &s) {
    blocks_.release_all(s.blocks);
    s.blocks.clear();
    if (draft_blocks_) draft_blocks_->release_all(s.draft_blocks);
    s.draft_blocks.clear();
    s.draft_computed = 0;
    s.computed = 0;
    s.sealed = 0;
    s.chain = 0;
    stats_.preemptions++;
}

void Engine::finish(Sequence &s, FinishReason reason) {
    Delta d;
    d.tokens = std::move(s.pending);
    s.pending.clear();
    d.finished = true;
    d.reason = reason;
    d.prompt_tokens = s.prompt_len;
    d.cached_tokens = s.cached;
    d.generated = s.generated;
    d.first_token_ms = s.first_token_ms;
    s.stream->push(d);
    blocks_.release_all(s.blocks);
    s.blocks.clear();
    if (draft_blocks_) draft_blocks_->release_all(s.draft_blocks);
    s.draft_blocks.clear();
    stats_.requests_done++;
}

// Builds the batch. Called with mutex_ held.
bool Engine::schedule() {
    batch_.clear();
    batch_seqs_.clear();
    batch_counts_.clear();
    int budget = opt_.max_batch_tokens;

    // Drop cancelled requests first: their blocks are needed.
    auto cancelled = [&](std::vector<SeqPtr> &v) {
        for (size_t i = 0; i < v.size();) {
            if (v[i]->stream->cancelled()) {
                finish(*v[i], FinishReason::cancelled);
                v.erase(v.begin() + static_cast<ptrdiff_t>(i));
            } else {
                i++;
            }
        }
    };
    cancelled(running_);
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        if ((*it)->stream->cancelled()) {
            finish(**it, FinishReason::cancelled);
            it = waiting_.erase(it);
        } else {
            ++it;
        }
    }

    auto add = [&](Sequence &s, int n) {
        for (int i = 0; i < n; i++) {
            batch_.tokens.push_back(s.tokens[static_cast<size_t>(s.computed + i)]);
            batch_.positions.push_back(s.computed + i);
        }
        batch_.query_start.push_back(static_cast<int32_t>(batch_.tokens.size()));
        batch_.context_len.push_back(s.computed + n);
        batch_seqs_.push_back(&s);
        batch_counts_.push_back(n);
        budget -= n;
    };

    // 1 and 2: running sequences, decodes before prompt chunks.
    bool preempted = false;
    for (int pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < running_.size() && budget > 0;) {
            Sequence &s = *running_[i];
            int need = static_cast<int>(s.tokens.size()) - s.computed;
            bool decoding = need == 1;
            if (decoding != (pass == 0) || need <= 0) {
                i++;
                continue;
            }
            int n = std::min({need, budget, opt_.max_prefill_chunk});
            // Make room, preempting the newest sequences (from the back) if needed.
            bool ok = ensure_blocks(s, s.computed + n);
            while (!ok) {
                Sequence &victim = *running_.back();
                if (&victim == &s) break;
                auto in_batch = std::find(batch_seqs_.begin(), batch_seqs_.end(), &victim);
                if (in_batch != batch_seqs_.end()) break;  // never preempt what is already in this step
                preempt(victim);
                waiting_.push_front(std::move(running_.back()));
                running_.pop_back();
                preempted = true;
                ok = ensure_blocks(s, s.computed + n);
            }
            if (!ok) {
                preempt(s);
                waiting_.push_front(std::move(running_[i]));
                running_.erase(running_.begin() + static_cast<ptrdiff_t>(i));
                preempted = true;
                continue;
            }
            add(s, n);
            i++;
        }
    }

    // 3: admit waiting requests while there is room (not right after a preemption).
    while (!preempted && !waiting_.empty() && budget > 0 && static_cast<int>(running_.size()) < opt_.max_seqs) {
        Sequence &s = *waiting_.front();
        std::vector<int> matched;
        if (s.computed == 0 && s.blocks.empty()) {
            // Reuse cached blocks, but always compute at least the last token: its logits are needed.
            matched = blocks_.match_prefix(std::span<const int32_t>(s.tokens).first(s.tokens.size() - 1));
            s.blocks = matched;
            s.computed = static_cast<int>(matched.size()) * kBlockSize;
            s.sealed = static_cast<int>(matched.size());
            s.chain = 0;
            for (int i = 0; i < s.sealed; i++)
                s.chain = BlockManager::block_hash(
                    s.chain, std::span<const int32_t>(s.tokens).subspan(static_cast<size_t>(i) * kBlockSize, kBlockSize));
        }
        int need = static_cast<int>(s.tokens.size()) - s.computed;
        int n = std::min({need, budget, opt_.max_prefill_chunk});
        if (!ensure_blocks(s, s.computed + n)) {
            // Not enough memory for even this chunk: give back what was taken and wait.
            blocks_.release_all(s.blocks);
            s.blocks.clear();
            s.computed = 0;
            s.sealed = 0;
            break;
        }
        if (!s.admitted_once) {
            s.cached = s.computed;
            stats_.cached_tokens += static_cast<uint64_t>(s.computed);
            s.admitted_once = true;
        }
        add(s, n);
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }

    if (batch_seqs_.empty()) return false;

    // Block tables, padded to the longest.
    int max_blocks = 0;
    for (Sequence *s : batch_seqs_) max_blocks = std::max(max_blocks, static_cast<int>(s->blocks.size()));
    batch_.max_blocks = max_blocks;
    batch_.block_tables.assign(batch_seqs_.size() * static_cast<size_t>(max_blocks), 0);
    sample_reqs_.clear();
    for (size_t i = 0; i < batch_seqs_.size(); i++) {
        Sequence &s = *batch_seqs_[i];
        std::copy(s.blocks.begin(), s.blocks.end(), batch_.block_tables.begin() + static_cast<ptrdiff_t>(i * max_blocks));
        if (s.computed + batch_counts_[i] == static_cast<int>(s.tokens.size())) {
            batch_.logit_rows.push_back(batch_.query_start[i + 1] - 1);
            SampleRequest r;
            r.params = s.req.sampling;
            r.counter = s.tokens.size();
            sample_reqs_.push_back(r);
        }
    }
    return true;
}

bool Engine::step() {
    auto t0 = Clock::now();
    bool spec;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        take_submissions();
        spec = spec_possible();
    }
    if (spec && spec_step()) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        stats_.steps++;
        const double a = 0.05;
        auto ema = [&](double &v, double x) { v = stats_.steps == 1 ? x : (1 - a) * v + a * x; };
        ema(stats_.step_ms, ms);
        ema(stats_.tokens_per_second, ms > 0 ? last_new_tokens_ / (ms / 1000.0) : 0);
        ema(stats_.prefill_per_second, 0);
        stats_.last_batch_seqs = static_cast<double>(running_.size());
        stats_.last_batch_tokens = batch_.num_tokens();
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!schedule()) return false;
    }

    backend_.forward(batch_);
    sampled_.resize(sample_reqs_.size());
    backend_.sample(sample_reqs_, sampled_);

    std::lock_guard<std::mutex> lock(mutex_);
    const double now_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    size_t row = 0;
    int new_tokens = 0, prompt_computed = 0;
    std::vector<Sequence *> done;
    std::vector<double> first_tokens;
    std::vector<FinishReason> reasons;
    for (size_t i = 0; i < batch_seqs_.size(); i++) {
        Sequence &s = *batch_seqs_[i];
        const int n = batch_counts_[i];
        if (s.computed < s.prompt_len) prompt_computed += std::min(n, s.prompt_len - s.computed);
        s.computed += n;
        seal_full_blocks(s);
        if (s.computed != static_cast<int>(s.tokens.size())) continue;  // prompt not finished yet

        const int32_t tok = sampled_[row++];
        s.tokens.push_back(tok);
        s.generated++;
        new_tokens++;
        if (s.generated == 1) {
            s.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - s.arrival).count();
            first_tokens.push_back(s.first_token_ms);
        }
        FinishReason reason = FinishReason::none;
        bool is_eos = std::find(eos_.begin(), eos_.end(), tok) != eos_.end() ||
                      std::find(s.req.stop_tokens.begin(), s.req.stop_tokens.end(), tok) != s.req.stop_tokens.end();
        if (is_eos && !s.req.ignore_eos) reason = FinishReason::stop;
        else if (s.generated >= s.req.max_tokens || static_cast<int>(s.tokens.size()) >= opt_.max_context) reason = FinishReason::length;
        if (reason != FinishReason::stop || s.req.ignore_eos) s.pending.push_back(tok);  // the stop token itself is not text
        if (reason != FinishReason::none) {
            done.push_back(&s);
            reasons.push_back(reason);
        } else {
            Delta d;
            d.tokens = std::move(s.pending);
            s.pending.clear();
            d.prompt_tokens = s.prompt_len;
            d.cached_tokens = s.cached;
            d.generated = s.generated;
            d.first_token_ms = s.generated == 1 ? s.first_token_ms : 0;
            s.stream->push(d);
        }
    }
    for (size_t i = 0; i < done.size(); i++) {
        finish(*done[i], reasons[i]);
        auto it = std::find_if(running_.begin(), running_.end(), [&](const SeqPtr &p) { return p.get() == done[i]; });
        if (it != running_.end()) running_.erase(it);
    }

    // Statistics: exponential moving averages over recent steps.
    stats_.steps++;
    stats_.generated_tokens += static_cast<uint64_t>(new_tokens);
    stats_.prompt_tokens += static_cast<uint64_t>(prompt_computed);
    const double a = 0.05, secs = now_ms / 1000.0;
    auto ema = [&](double &v, double x) { v = stats_.steps == 1 ? x : (1 - a) * v + a * x; };
    ema(stats_.step_ms, now_ms);
    ema(stats_.tokens_per_second, secs > 0 ? new_tokens / secs : 0);
    ema(stats_.prefill_per_second, secs > 0 ? prompt_computed / secs : 0);
    stats_.last_batch_tokens = batch_.num_tokens();
    stats_.last_batch_seqs = batch_.num_seqs();
    // (batch_seqs_ may point to sequences finished and freed above: use the copied values)
    for (double ms : first_tokens) ema(stats_.ttft_ms_avg, ms);
    return true;
}

// ---- speculative decoding ---------------------------------------------------------------
//
// The draft model proposes k tokens per sequence; the target checks all of them
// in one forward pass over k + 1 positions. Both models sample with the same
// Gumbel noise for a given (seed, position) - the noise is a pure function of
// them - so the target's sample at each position is exactly what it would have
// drawn without speculation. Draft tokens are accepted while they equal the
// target's samples; at the first difference the target's own token is used.
// The output is therefore identical to plain decoding with the same seed; only
// the number of target passes changes.

bool Engine::spec_possible() const {
    if (!draft_ || running_.empty() || !waiting_.empty() || !incoming_.empty()) return false;
    const int verify_rows = static_cast<int>(running_.size()) * (opt_.spec_tokens + 1);
    if (verify_rows > opt_.max_batch_tokens || verify_rows > backend_.options().max_logit_rows) return false;
    for (const auto &s : running_)
        if (static_cast<int>(s->tokens.size()) - s->computed != 1) return false;  // someone is still reading its prompt
    return true;
}

bool Engine::ensure_draft_blocks(Sequence &s, int upto) {
    const size_t need = static_cast<size_t>((upto + kBlockSize - 1) / kBlockSize);
    while (s.draft_blocks.size() < need) {
        int b = draft_blocks_->allocate();
        if (b < 0) return false;
        s.draft_blocks.push_back(b);
    }
    return true;
}

// Runs one forward pass of the draft model over `parts` (sequence, first token
// index, count) of each sequence's extended token list, with logits for the
// sequences in `want_logits`; returns the sampled tokens for those.
std::vector<int32_t> Engine::draft_forward(const std::vector<Sequence *> &seqs, const std::vector<std::vector<int32_t>> &ext,
                                           const std::vector<std::pair<int, int>> &range, bool sample) {
    StepBatch &b = draft_batch_;
    b.clear();
    std::vector<SampleRequest> reqs;
    int max_blocks = 0;
    std::vector<int> order;
    for (size_t i = 0; i < seqs.size(); i++) {
        const auto [from, count] = range[i];
        if (count <= 0) continue;
        order.push_back(static_cast<int>(i));
        for (int p = from; p < from + count; p++) {
            b.tokens.push_back(ext[i][static_cast<size_t>(p)]);
            b.positions.push_back(p);
        }
        b.query_start.push_back(static_cast<int32_t>(b.tokens.size()));
        b.context_len.push_back(from + count);
        max_blocks = std::max(max_blocks, static_cast<int>(seqs[i]->draft_blocks.size()));
        if (sample) {
            b.logit_rows.push_back(static_cast<int32_t>(b.tokens.size()) - 1);
            SampleRequest r;
            r.params = seqs[i]->req.sampling;
            r.counter = static_cast<uint64_t>(from + count);  // the position being predicted
            reqs.push_back(r);
        }
    }
    if (order.empty()) return {};
    b.max_blocks = max_blocks;
    b.block_tables.assign(order.size() * static_cast<size_t>(max_blocks), 0);
    for (size_t j = 0; j < order.size(); j++) {
        const auto &bl = seqs[static_cast<size_t>(order[j])]->draft_blocks;
        std::copy(bl.begin(), bl.end(), b.block_tables.begin() + static_cast<ptrdiff_t>(j * max_blocks));
    }
    draft_->forward(b);
    std::vector<int32_t> out(reqs.size());
    if (sample) draft_->sample(reqs, out);
    return out;
}

bool Engine::spec_step() {
    const int k = opt_.spec_tokens;
    std::vector<Sequence *> seqs;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &s : running_) {
            // Target cache room for the k + 1 verified positions.
            if (!ensure_blocks(*s, s->computed + k + 1)) return false;
            seqs.push_back(s.get());
        }
    }
    const size_t S = seqs.size();
    std::vector<std::vector<int32_t>> ext(S);
    for (size_t i = 0; i < S; i++) ext[i] = seqs[i]->tokens;

    // 1. Bring the draft's cache up to date (everything but the last token), in budget-sized chunks.
    for (;;) {
        std::vector<std::pair<int, int>> range(S, {0, 0});
        int budget = opt_.max_batch_tokens;
        bool any = false;
        for (size_t i = 0; i < S && budget > 0; i++) {
            Sequence &s = *seqs[i];
            int missing = static_cast<int>(ext[i].size()) - 1 - s.draft_computed;
            if (missing <= 0) continue;
            int n = std::min(missing, budget);
            if (!ensure_draft_blocks(s, s.draft_computed + n)) return false;
            range[i] = {s.draft_computed, n};
            budget -= n;
            any = true;
        }
        if (!any) break;
        draft_forward(seqs, ext, range, false);
        for (size_t i = 0; i < S; i++) seqs[i]->draft_computed += range[i].second;
    }

    // 2. k proposals: one token per sequence per pass (pure decode, so CUDA graphs apply).
    for (int j = 0; j < k; j++) {
        std::vector<std::pair<int, int>> range(S);
        for (size_t i = 0; i < S; i++) {
            Sequence &s = *seqs[i];
            if (!ensure_draft_blocks(s, static_cast<int>(ext[i].size()))) return false;
            range[i] = {static_cast<int>(ext[i].size()) - 1, 1};
        }
        auto toks = draft_forward(seqs, ext, range, true);
        for (size_t i = 0; i < S; i++) {
            ext[i].push_back(toks[i]);
            seqs[i]->draft_computed = static_cast<int>(ext[i].size()) - 1;
        }
    }

    // 3. Verify: the target reads [last token, d1..dk] and samples at every position.
    StepBatch &b = batch_;
    b.clear();
    sample_reqs_.clear();
    int max_blocks = 0;
    for (size_t i = 0; i < S; i++) {
        Sequence &s = *seqs[i];
        const int n = static_cast<int>(s.tokens.size());
        for (int p = n - 1; p <= n - 1 + k; p++) {
            b.tokens.push_back(ext[i][static_cast<size_t>(p)]);
            b.positions.push_back(p);
            b.logit_rows.push_back(static_cast<int32_t>(b.tokens.size()) - 1);
            SampleRequest r;
            r.params = s.req.sampling;
            r.counter = static_cast<uint64_t>(p + 1);
            sample_reqs_.push_back(r);
        }
        b.query_start.push_back(static_cast<int32_t>(b.tokens.size()));
        b.context_len.push_back(n + k);
        max_blocks = std::max(max_blocks, static_cast<int>(s.blocks.size()));
    }
    b.max_blocks = max_blocks;
    b.block_tables.assign(S * static_cast<size_t>(max_blocks), 0);
    for (size_t i = 0; i < S; i++)
        std::copy(seqs[i]->blocks.begin(), seqs[i]->blocks.end(), b.block_tables.begin() + static_cast<ptrdiff_t>(i * max_blocks));
    backend_.forward(b);
    sampled_.resize(sample_reqs_.size());
    backend_.sample(sample_reqs_, sampled_);

    // 4. Accept, append, stream.
    std::lock_guard<std::mutex> lock(mutex_);
    int new_tokens = 0;
    std::vector<Sequence *> done;
    std::vector<FinishReason> reasons;
    for (size_t i = 0; i < S; i++) {
        Sequence &s = *seqs[i];
        const int32_t *t = &sampled_[i * static_cast<size_t>(k + 1)];
        const int n = static_cast<int>(s.tokens.size());
        int m = 0;
        while (m < k && t[m] == ext[i][static_cast<size_t>(n + m)]) m++;
        stats_.spec_proposed += static_cast<uint64_t>(k);
        stats_.spec_accepted += static_cast<uint64_t>(m);
        FinishReason reason = FinishReason::none;
        int appended = 0;
        for (int j = 0; j <= m && reason == FinishReason::none; j++) {
            const int32_t tok = t[j];
            s.tokens.push_back(tok);
            s.generated++;
            appended++;
            new_tokens++;
            if (s.generated == 1) s.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - s.arrival).count();
            bool is_eos = std::find(eos_.begin(), eos_.end(), tok) != eos_.end() ||
                          std::find(s.req.stop_tokens.begin(), s.req.stop_tokens.end(), tok) != s.req.stop_tokens.end();
            if (is_eos && !s.req.ignore_eos) reason = FinishReason::stop;
            else if (s.generated >= s.req.max_tokens || static_cast<int>(s.tokens.size()) >= opt_.max_context) reason = FinishReason::length;
            if (reason != FinishReason::stop || s.req.ignore_eos) s.pending.push_back(tok);
        }
        // Cache entries are valid for the inputs that matched: positions n - 1 .. n - 1 + m.
        s.computed = n + std::min(m, appended - 1);
        s.draft_computed = std::min(s.draft_computed, s.computed);
        seal_full_blocks(s);
        if (reason != FinishReason::none) {
            done.push_back(&s);
            reasons.push_back(reason);
        } else {
            Delta d;
            d.tokens = std::move(s.pending);
            s.pending.clear();
            d.prompt_tokens = s.prompt_len;
            d.cached_tokens = s.cached;
            d.generated = s.generated;
            d.first_token_ms = s.generated == appended ? s.first_token_ms : 0;
            s.stream->push(d);
        }
    }
    for (size_t i = 0; i < done.size(); i++) {
        finish(*done[i], reasons[i]);
        auto it = std::find_if(running_.begin(), running_.end(), [&](const SeqPtr &p) { return p.get() == done[i]; });
        if (it != running_.end()) running_.erase(it);
    }
    stats_.spec_steps++;
    stats_.generated_tokens += static_cast<uint64_t>(new_tokens);
    last_new_tokens_ = new_tokens;
    return true;
}

std::vector<int32_t> Engine::generate(Request req, Delta *final_delta) {
    EMBER_CHECK(!started_, "generate() runs the engine on the calling thread; stop() the loop first");
    auto stream = submit(std::move(req));
    std::vector<int32_t> out;
    Delta d;
    for (;;) {
        step();
        while (stream->next(d, std::chrono::milliseconds(0))) {
            out.insert(out.end(), d.tokens.begin(), d.tokens.end());
            if (d.finished) {
                if (final_delta) *final_delta = d;
                return out;
            }
        }
    }
}

EngineStats Engine::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    EngineStats s = stats_;
    s.running = static_cast<int>(running_.size());
    s.waiting = static_cast<int>(waiting_.size() + incoming_.size());
    s.kv_used = blocks_.used();
    s.kv_cached = blocks_.cached();
    s.uptime_s = std::chrono::duration<double>(Clock::now() - start_time_).count();
    return s;
}

std::vector<RequestInfo> Engine::requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<RequestInfo> out;
    auto now = Clock::now();
    auto describe = [&](const Sequence &s, const char *state) {
        out.push_back({s.id, s.req.tag, state, s.prompt_len, s.cached, s.generated, s.req.max_tokens,
                       static_cast<int>(s.blocks.size()), std::chrono::duration<double, std::milli>(now - s.arrival).count()});
    };
    for (const auto &s : running_) describe(*s, s->computed < s->prompt_len ? "prefill" : "decode");
    for (const auto &s : waiting_) describe(*s, "waiting");
    for (const auto &s : incoming_) describe(*s, "waiting");
    return out;
}

std::vector<uint8_t> Engine::kv_block_states() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.states();
}

std::string Engine::check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string problem = blocks_.check();
    if (!problem.empty()) return problem;
    // Every referenced block is held by some sequence, as many times as it is referenced.
    std::vector<int> refs(static_cast<size_t>(blocks_.num_blocks()), 0);
    for (const auto &s : running_)
        for (int b : s->blocks) refs[static_cast<size_t>(b)]++;
    for (const auto &s : waiting_)
        for (int b : s->blocks) refs[static_cast<size_t>(b)]++;
    for (int b = 0; b < blocks_.num_blocks(); b++)
        if (refs[static_cast<size_t>(b)] != blocks_.refcount(b))
            return std::format("block {} has {} references but {} holders", b, blocks_.refcount(b), refs[static_cast<size_t>(b)]);
    return {};
}

}  // namespace ember
