// A fixed pool of worker threads for data-parallel loops (the CPU backend's
// matrix kernels). parallel_for blocks until every chunk is done; the calling
// thread works too, so a pool of N threads uses N cores, not N + 1.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ember {

class ThreadPool {
public:
    explicit ThreadPool(int threads = 0);  // 0 = hardware concurrency
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }

    // Calls fn(begin, end) over [0, n) split into chunks of at least `grain`.
    // One caller at a time (the engine thread); fn must not call parallel_for.
    void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)> &fn);

private:
    void worker_loop();
    void run_chunks();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(size_t, size_t)> *job_ = nullptr;
    size_t n_ = 0, chunk_ = 1, next_ = 0;
    size_t finished_ = 0;  // chunks of the current job that have finished
    unsigned generation_ = 0;
    bool stop_ = false;
};

}  // namespace ember
