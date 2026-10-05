// The engine: continuous batching over a backend.
//
// Requests arrive at any time (submit() is thread-safe). Every step the
// scheduler builds one ragged batch:
//   1. one token for every sequence that is generating (decode first, so
//      generation never stalls behind a long prompt);
//   2. chunks of prompts still being read (chunked prefill), within a token
//      budget per step;
//   3. new requests, as long as budget and KV blocks allow.
// It runs the model once, samples, and streams the new tokens to each
// request. When the KV cache is full, the most recently started sequence is
// preempted: its blocks are released (the full ones stay in the prefix cache)
// and it is recomputed later.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "backend/backend.hpp"
#include "kvcache/block_manager.hpp"

namespace ember {

enum class FinishReason { none, stop, length, cancelled, error };
const char *finish_reason_name(FinishReason r);

struct Request {
    std::vector<int32_t> prompt;
    SamplingParams sampling;
    int max_tokens = 256;
    bool ignore_eos = false;
    std::vector<int32_t> stop_tokens;  // in addition to the model's end-of-sequence tokens
    std::string tag;                   // free text shown on the dashboard (e.g. the API route)
};

// What a request receives, a few tokens at a time.
struct Delta {
    std::vector<int32_t> tokens;
    bool finished = false;
    FinishReason reason = FinishReason::none;
    int prompt_tokens = 0;
    int cached_tokens = 0;   // prompt tokens served from the prefix cache
    int generated = 0;       // total generated so far
    double first_token_ms = 0;  // time from submission to the first token
};

class RequestStream {
public:
    explicit RequestStream(uint64_t id) : id_(id) {}
    uint64_t id() const { return id_; }

    // Consumer side: waits up to `timeout` for news; merges everything pending.
    // Returns false on timeout.
    bool next(Delta &out, std::chrono::milliseconds timeout);
    void cancel() { cancelled_.store(true); }
    bool cancelled() const { return cancelled_.load(); }

    // Engine side.
    void push(const Delta &d);

private:
    uint64_t id_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Delta> queue_;
    std::atomic<bool> cancelled_{false};
};

struct EngineOptions {
    int max_batch_tokens = 0;      // 0 = the backend's limit
    int max_seqs = 0;              // 0 = the backend's limit
    int max_prefill_chunk = 512;   // prompt tokens per sequence per step
    bool prefix_caching = true;
    int max_context = 0;           // 0 = the model's max_position
};

struct RequestInfo {
    uint64_t id;
    std::string tag;
    std::string state;  // waiting, prefill, decode
    int prompt_tokens, cached_tokens, generated, max_tokens, blocks;
    double age_ms;
};

struct EngineStats {
    uint64_t steps = 0, requests_done = 0, prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0, preemptions = 0;
    int running = 0, waiting = 0;
    int kv_blocks = 0, kv_used = 0, kv_cached = 0;
    double tokens_per_second = 0;     // generated tokens, recent average
    double prefill_per_second = 0;    // prompt tokens computed, recent average
    double step_ms = 0;               // recent average step time
    double last_batch_tokens = 0, last_batch_seqs = 0;
    double ttft_ms_avg = 0;           // time to first token, recent average
    double uptime_s = 0;
};

class Engine {
public:
    Engine(Backend &backend, const EngineOptions &options);
    ~Engine();

    std::shared_ptr<RequestStream> submit(Request request);

    // Runs the loop on a background thread.
    void start();
    void stop();

    // One scheduling step on the calling thread (when not started); false if idle.
    bool step();

    // Runs a request to completion on the calling thread (no background loop).
    std::vector<int32_t> generate(Request request, Delta *final_delta = nullptr);

    EngineStats stats() const;
    std::vector<RequestInfo> requests() const;
    std::vector<uint8_t> kv_block_states() const;
    std::string check() const;  // block accounting audit; empty if consistent

    Backend &backend() { return backend_; }
    const EngineOptions &options() const { return opt_; }

private:
    struct Sequence;
    using SeqPtr = std::unique_ptr<Sequence>;

    void loop();
    void take_submissions();
    bool schedule();
    void finish(Sequence &s, FinishReason reason);
    void preempt(Sequence &s);
    bool ensure_blocks(Sequence &s, int upto_tokens);
    void seal_full_blocks(Sequence &s);

    Backend &backend_;
    EngineOptions opt_;
    BlockManager blocks_;
    std::vector<int> eos_;

    mutable std::mutex mutex_;          // guards incoming_, the sequence lists and stats_
    std::condition_variable wake_;
    std::vector<SeqPtr> incoming_;
    std::deque<SeqPtr> waiting_;
    std::vector<SeqPtr> running_;
    uint64_t next_id_ = 1;

    std::thread thread_;
    std::atomic<bool> stop_{false};
    bool started_ = false;

    StepBatch batch_;
    std::vector<Sequence *> batch_seqs_;
    std::vector<int> batch_counts_;
    std::vector<SampleRequest> sample_reqs_;
    std::vector<int32_t> sampled_;

    EngineStats stats_;
    std::chrono::steady_clock::time_point start_time_;
};

}  // namespace ember
