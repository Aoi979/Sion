#pragma once

#include "cuda_utils.hpp"
#include "stats.hpp"

#include <chrono>
#include <functional>
#include <vector>

namespace sion::bench {

struct TimingConfig {
  int warmup = 10;
  int repeat = 30;
  int iters = 0;
  int max_iters = 100000;
  double min_sample_ms = 10.0;
};

struct TimingResult {
  int iters = 1;
  SeriesStats gpu_ms;
  SeriesStats host_issue_us;
  SeriesStats e2e_us;
};

using LaunchFn = std::function<void(cudaStream_t)>;

inline double gpu_elapsed_ms_once(const LaunchFn &fn, cudaStream_t stream,
                                  int iters) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
  for (int i = 0; i < iters; ++i) {
    fn(stream);
  }
  cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  return static_cast<double>(ms);
}

inline int choose_iters(const LaunchFn &fn, cudaStream_t stream,
                        const TimingConfig &cfg) {
  if (cfg.iters > 0) {
    return cfg.iters;
  }
  int iters = 1;
  while (iters < cfg.max_iters) {
    const double ms = gpu_elapsed_ms_once(fn, stream, iters);
    if (ms >= cfg.min_sample_ms) {
      break;
    }
    iters *= 2;
  }
  return iters;
}

inline TimingResult run_timing(const LaunchFn &fn, const TimingConfig &cfg,
                               cudaStream_t stream) {
  for (int w = 0; w < cfg.warmup; ++w) {
    fn(stream);
  }
  cuda_check(cudaStreamSynchronize(stream), "warmup cudaStreamSynchronize");

  TimingResult result;
  result.iters = choose_iters(fn, stream, cfg);
  cuda_check(cudaStreamSynchronize(stream), "choose_iters cudaStreamSynchronize");

  std::vector<double> gpu_ms;
  std::vector<double> host_issue_us;
  std::vector<double> e2e_us;
  gpu_ms.reserve(cfg.repeat);
  host_issue_us.reserve(cfg.repeat);
  e2e_us.reserve(cfg.repeat);

  for (int r = 0; r < cfg.repeat; ++r) {
    const double ms = gpu_elapsed_ms_once(fn, stream, result.iters);
    gpu_ms.push_back(ms / static_cast<double>(result.iters));
  }

  for (int r = 0; r < cfg.repeat; ++r) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < result.iters; ++i) {
      fn(stream);
    }
    auto stop = std::chrono::steady_clock::now();
    const auto us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    host_issue_us.push_back(us / static_cast<double>(result.iters));
    cuda_check(cudaStreamSynchronize(stream),
               "host issue cudaStreamSynchronize");
  }

  for (int r = 0; r < cfg.repeat; ++r) {
    auto start = std::chrono::steady_clock::now();
    fn(stream);
    cuda_check(cudaStreamSynchronize(stream), "e2e cudaStreamSynchronize");
    auto stop = std::chrono::steady_clock::now();
    e2e_us.push_back(
        std::chrono::duration<double, std::micro>(stop - start).count());
  }

  result.gpu_ms = summarize(std::move(gpu_ms));
  result.host_issue_us = summarize(std::move(host_issue_us));
  result.e2e_us = summarize(std::move(e2e_us));
  return result;
}

} // namespace sion::bench
