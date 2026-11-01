#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "brave_rt/denormals.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace brave_rt {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

/**
 * Fork-join pool for intra-layer parallelism on the audio thread.
 *
 * run(f) calls f(tid) on every thread (caller is tid 0) and returns when all
 * finish. Work is split statically by tid, so a given worker always computes
 * the same weight panels → its slice of the weights stays hot in its own L2.
 *
 * Between layers inside a tick, workers spin (sub-µs handoff). After
 * kSpinIters of idleness (≈ between ticks / host stopped) they park on a
 * condvar so they do not burn cores; the first layer of the next tick then
 * pays one futex wake. run() does not allocate.
 */
class SpinPool {
 public:
  explicit SpinPool(int n_threads) : n_(n_threads < 1 ? 1 : n_threads) {
    for (int t = 1; t < n_; ++t) workers_.emplace_back([this, t] { loop(t); });
  }

  ~SpinPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_.store(true);
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  SpinPool(const SpinPool&) = delete;
  SpinPool& operator=(const SpinPool&) = delete;

  int size() const { return n_; }

  template <class F>
  void run(F& f) {
    if (n_ == 1) {
      f(0);
      return;
    }
    ctx_ = &f;
    fn_ = [](void* c, int tid) { (*static_cast<F*>(c))(tid); };
    pending_.store(n_ - 1);
    gen_.fetch_add(1);  // seq_cst: pairs with sleepers_ check below
    if (sleepers_.load() > 0) {
      std::lock_guard<std::mutex> lk(m_);
      cv_.notify_all();
    }
    f(0);
    while (pending_.load(std::memory_order_acquire) != 0) cpu_relax();
  }

  /** Static contiguous split of [0, n) for thread tid. */
  static void split(int n, int n_threads, int tid, int& begin, int& end) {
    begin = static_cast<int>(static_cast<int64_t>(n) * tid / n_threads);
    end = static_cast<int>(static_cast<int64_t>(n) * (tid + 1) / n_threads);
  }

 private:
  static constexpr int kSpinIters = 200000;  // ~ a few hundred µs of pause

  void loop(int tid) {
    ScopedNoDenormals no_denormals;  // MXCSR/FPCR are per-thread
    uint64_t seen = 0;
    for (;;) {
      int spins = 0;
      while (gen_.load(std::memory_order_acquire) == seen) {
        if (stop_.load()) return;
        if (++spins < kSpinIters) {
          cpu_relax();
          continue;
        }
        std::unique_lock<std::mutex> lk(m_);
        sleepers_.fetch_add(1);
        cv_.wait(lk, [&] { return gen_.load() != seen || stop_.load(); });
        sleepers_.fetch_sub(1);
        spins = 0;
      }
      if (stop_.load()) return;
      seen = gen_.load(std::memory_order_acquire);
      fn_(ctx_, tid);
      pending_.fetch_sub(1, std::memory_order_release);
    }
  }

  const int n_;
  std::vector<std::thread> workers_;
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> pending_{0};
  std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  std::mutex m_;
  std::condition_variable cv_;
  void (*fn_)(void*, int) = nullptr;
  void* ctx_ = nullptr;
};

}  // namespace brave_rt
