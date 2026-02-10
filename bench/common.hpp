#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace sion::bench {

struct BenchmarkConfig {
  int warmup = 5;
  int repeat = 20;
  int iters = 10;
};

struct BenchmarkStats {
  double min_ms = 0.0;
  double max_ms = 0.0;
  double avg_ms = 0.0;
  double tflops = 0.0;
};

inline void cuda_check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "[bench] " << msg << ": " << cudaGetErrorString(err) << "\n";
    std::terminate();
  }
}

inline BenchmarkStats run_kernel_bench(std::function<void(cudaStream_t)> fn,
                                       const BenchmarkConfig &cfg,
                                       cudaStream_t stream) {
  for (int w = 0; w < cfg.warmup; ++w) {
    for (int i = 0; i < cfg.iters; ++i) {
      fn(stream);
    }
  }
  cuda_check(cudaStreamSynchronize(stream), "warmup sync failed");

  cudaEvent_t start{};
  cudaEvent_t stop{};
  cuda_check(cudaEventCreate(&start), "event create start failed");
  cuda_check(cudaEventCreate(&stop), "event create stop failed");

  std::vector<double> times;
  times.reserve(cfg.repeat);
  for (int r = 0; r < cfg.repeat; ++r) {
    cuda_check(cudaEventRecord(start, stream), "event record start failed");
    for (int i = 0; i < cfg.iters; ++i) {
      fn(stream);
    }
    cuda_check(cudaEventRecord(stop, stream), "event record stop failed");
    cuda_check(cudaEventSynchronize(stop), "event sync failed");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop),
               "event elapsed time failed");
    times.push_back(static_cast<double>(ms) / cfg.iters);
  }

  cuda_check(cudaEventDestroy(start), "event destroy start failed");
  cuda_check(cudaEventDestroy(stop), "event destroy stop failed");

  BenchmarkStats stats;
  stats.min_ms = *std::min_element(times.begin(), times.end());
  stats.max_ms = *std::max_element(times.begin(), times.end());
  stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) /
                 static_cast<double>(times.size());

  return stats;
}

inline void print_stats_md_file(const BenchmarkStats &stats,
                                const std::string &bench_name,
                                const std::string &device,
                                const std::string &dtype,
                                const std::string &shape,
                                const std::string &filename = "sion_bench.md",
                                bool header = false) {
  std::ofstream out(filename, header ? std::ios::trunc : std::ios::app);
  if (header) {
    out << "| Bench Name | Device | DType | Shape | min_ms | max_ms | avg_ms | "
           "tflops |\n";
    out << "|------------|--------|-------|-------|--------|--------|--------|"
           "--------|\n";
  }

  out << "| " << bench_name << " | " << device << " | " << dtype << " | "
      << shape << " | " << stats.min_ms << " | " << stats.max_ms << " | "
      << stats.avg_ms << " | " << stats.tflops << " |\n";
}

} // namespace sion::bench
