#include "common/thread_pool.hpp"

#include <algorithm>

namespace ember {

ThreadPool::ThreadPool(int threads) {
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 1; i < threads; i++) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto &t : workers_) t.join();
}

// Takes chunks of the current job until none are left. A chunk always belongs
// to the job that is current when it is taken, and that job cannot complete
// before the chunk does, so the counters never mix two jobs.
void ThreadPool::run_chunks() {
    for (;;) {
        const std::function<void(size_t, size_t)> *job;
        size_t begin, end;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!job_ || next_ >= n_) return;
            job = job_;
            begin = next_;
            end = std::min(n_, begin + chunk_);
            next_ = end;
        }
        (*job)(begin, end);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (++finished_ == (n_ + chunk_ - 1) / chunk_) done_.notify_all();
        }
    }
}

void ThreadPool::worker_loop() {
    unsigned seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        lock.unlock();
        run_chunks();
        lock.lock();
    }
}

void ThreadPool::parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)> &fn) {
    if (n == 0) return;
    grain = std::max<size_t>(grain, 1);
    size_t threads = static_cast<size_t>(size());
    if (workers_.empty() || n <= grain) {
        fn(0, n);
        return;
    }
    // About 4 chunks per thread balances uneven chunks without much locking.
    size_t chunk = std::max(grain, (n + threads * 4 - 1) / (threads * 4));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = &fn;
        n_ = n;
        chunk_ = chunk;
        next_ = 0;
        finished_ = 0;
        generation_++;
    }
    wake_.notify_all();
    run_chunks();
    std::unique_lock<std::mutex> lock(mutex_);
    size_t total = (n + chunk - 1) / chunk;
    done_.wait(lock, [&] { return finished_ == total; });
    job_ = nullptr;
}

}  // namespace ember
