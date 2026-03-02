#include "../common.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <felix/felix.hpp>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Shape {
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

struct Args {
  sion::bench::BenchmarkConfig bench_cfg;
  float alpha = 1.0f;
  float beta = 0.0f;
  std::string kernel = "cute_hgemm_128x128_nn";
  std::vector<Shape> shapes;
  std::string out = "hgemm_bench.md";
};

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog
      << " [--shape MxNxK] [--m M --n N --k K] [--alpha A --beta B]\n"
      << "       [--kernel NAME] [--warmup W --repeat R --iters I] [--out "
         "FILE]\n"
      << "Note: current HGEMM benchmark requires M/N % 128 == 0 and K % 64 "
         "== 0\n"
      << "Example:\n"
      << "  " << prog
      << " --shape 2048x2048x2048 --kernel cute_hgemm_128x128_nn "
         "--warmup 5 --repeat 20 --iters 10\n";
}

static bool parse_shape(const std::string &s, Shape &out) {
  char sep = (s.find('x') != std::string::npos) ? 'x' : ',';
  std::stringstream ss(s);
  std::string item;
  std::vector<uint32_t> vals;
  while (std::getline(ss, item, sep)) {
    if (item.empty()) {
      return false;
    }
    vals.push_back(static_cast<uint32_t>(std::stoul(item)));
  }
  if (vals.size() != 3) {
    return false;
  }
  out = {vals[0], vals[1], vals[2]};
  return true;
}

static std::string shape_to_string(const Shape &s) {
  std::ostringstream oss;
  oss << s.m << "x" << s.n << "x" << s.k;
  return oss.str();
}

static bool is_aligned_shape(const Shape &s) {
  return (s.m % 128u == 0u) && (s.n % 128u == 0u) && (s.k % 64u == 0u);
}

static void fill_random(std::vector<half> &v) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : v) {
    x = __float2half(dist(rng));
  }
}

int main(int argc, char **argv) {
  Args args;
  int m = -1, n = -1, k = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--shape" && i + 1 < argc) {
      Shape s{};
      if (!parse_shape(argv[++i], s)) {
        std::cerr << "Invalid --shape: " << argv[i] << "\n";
        return 1;
      }
      args.shapes.push_back(s);
    } else if (arg == "--m" && i + 1 < argc) {
      m = std::stoi(argv[++i]);
    } else if (arg == "--n" && i + 1 < argc) {
      n = std::stoi(argv[++i]);
    } else if (arg == "--k" && i + 1 < argc) {
      k = std::stoi(argv[++i]);
    } else if (arg == "--alpha" && i + 1 < argc) {
      args.alpha = std::stof(argv[++i]);
    } else if (arg == "--beta" && i + 1 < argc) {
      args.beta = std::stof(argv[++i]);
    } else if (arg == "--kernel" && i + 1 < argc) {
      args.kernel = argv[++i];
    } else if (arg == "--warmup" && i + 1 < argc) {
      args.bench_cfg.warmup = std::stoi(argv[++i]);
    } else if (arg == "--repeat" && i + 1 < argc) {
      args.bench_cfg.repeat = std::stoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      args.bench_cfg.iters = std::stoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      args.out = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown arg: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (m > 0 || n > 0 || k > 0) {
    if (m <= 0 || n <= 0 || k <= 0) {
      std::cerr << "If using --m/--n/--k, all must be > 0\n";
      return 1;
    }
    args.shapes.push_back(
        {static_cast<uint32_t>(m), static_cast<uint32_t>(n),
         static_cast<uint32_t>(k)});
  }

  if (args.shapes.empty()) {
    args.shapes.push_back({2048, 2048, 2048});
  }

  for (const auto &shape : args.shapes) {
    if (!is_aligned_shape(shape)) {
      std::cerr
          << "Unsupported shape " << shape_to_string(shape)
          << ": current HGEMM benchmark requires M/N multiple of 128 and K "
             "multiple of 64\n";
      return 1;
    }
  }

  cudaStream_t stream{};
  sion::bench::cuda_check(cudaStreamCreate(&stream),
                          "cuda stream create failed");

  cudaDeviceProp prop{};
  sion::bench::cuda_check(cudaGetDeviceProperties(&prop, 0),
                          "cuda get device properties failed");
  std::string device_name = prop.name;

  bool header = true;
  for (const auto &shape : args.shapes) {
    const size_t a_elems =
        static_cast<size_t>(shape.m) * static_cast<size_t>(shape.k);
    const size_t b_elems =
        static_cast<size_t>(shape.k) * static_cast<size_t>(shape.n);
    const size_t c_elems =
        static_cast<size_t>(shape.m) * static_cast<size_t>(shape.n);

    std::vector<half> hA(a_elems);
    std::vector<half> hB(b_elems);
    std::vector<half> hC(c_elems);
    fill_random(hA);
    fill_random(hB);
    fill_random(hC);

    half *dA = nullptr;
    half *dB = nullptr;
    half *dC = nullptr;
    sion::bench::cuda_check(cudaMalloc(&dA, a_elems * sizeof(half)),
                            "cudaMalloc A failed");
    sion::bench::cuda_check(cudaMalloc(&dB, b_elems * sizeof(half)),
                            "cudaMalloc B failed");
    sion::bench::cuda_check(cudaMalloc(&dC, c_elems * sizeof(half)),
                            "cudaMalloc C failed");

    sion::bench::cuda_check(
        cudaMemcpyAsync(dA, hA.data(), a_elems * sizeof(half),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync A failed");
    sion::bench::cuda_check(
        cudaMemcpyAsync(dB, hB.data(), b_elems * sizeof(half),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync B failed");
    sion::bench::cuda_check(
        cudaMemcpyAsync(dC, hC.data(), c_elems * sizeof(half),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync C failed");
    sion::bench::cuda_check(cudaStreamSynchronize(stream),
                            "H2D sync failed");

    auto launch = [&](cudaStream_t s) {
      auto status = felix::ampere_hgemm_launch(
          shape.m, shape.n, shape.k, args.alpha, dA, dB, args.beta, dC, s,
          args.kernel);
      if (!status.ok()) {
        std::cerr << "Kernel launch failed: " << status.str() << "\n";
        std::exit(1);
      }
    };

    auto stats = sion::bench::run_kernel_bench(launch, args.bench_cfg, stream);
    const double flops =
        2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
        static_cast<double>(shape.k);
    const double tflops = flops / (stats.avg_ms * 1e-3) / 1.0e12;
    stats.tflops = tflops;

    std::cout << "[Sion] hgemm " << shape_to_string(shape)
              << " avg_ms=" << stats.avg_ms << " tflops=" << stats.tflops
              << " kernel=" << args.kernel << "\n";

    sion::bench::print_stats_md_file(stats, args.kernel, device_name, "f16",
                                     shape_to_string(shape), args.out, header);
    header = false;

    sion::bench::cuda_check(cudaFree(dA), "cudaFree A failed");
    sion::bench::cuda_check(cudaFree(dB), "cudaFree B failed");
    sion::bench::cuda_check(cudaFree(dC), "cudaFree C failed");
  }

  sion::bench::cuda_check(cudaStreamDestroy(stream),
                          "cuda stream destroy failed");
  std::cout << "[Sion] benchmarks finished, results in " << args.out << "\n";
  return 0;
}
