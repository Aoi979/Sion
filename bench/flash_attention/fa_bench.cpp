#include "../common.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <felix/felix.hpp>
#include <torch/torch.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Shape {
  int64_t b;
  int64_t h;
  int64_t n;
  int64_t d;
};

struct Args {
  sion::bench::BenchmarkConfig bench_cfg;
  std::vector<Shape> shapes;
  std::string out = "fa_compare.md";
  std::string ref = "libtorch_sdpa";
  std::string kernel = "ampere_flash_attn_mma16168_64_1D_warp_tiling";
};

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog
      << " [--shape BxHxNxD] [--b B --h H --n N --d D]\n"
      << "       [--ref libtorch_sdpa|none] [--kernel NAME]\n"
      << "       [--warmup W --repeat R --iters I]\n"
      << "       [--out FILE]\n"
      << "Note: seq_len(N) must be divisible by 64; head_dim(D) supports 64 "
         "or 128.\n"
      << "Example:\n"
      << "  " << prog
      << " --shape 8x16x1024x64 --shape 8x16x2048x128 --out fa_compare.md\n";
}

static bool parse_shape(const std::string &s, Shape &out) {
  char sep = (s.find('x') != std::string::npos) ? 'x' : ',';
  std::stringstream ss(s);
  std::string item;
  std::vector<int64_t> vals;
  while (std::getline(ss, item, sep)) {
    if (item.empty()) {
      return false;
    }
    vals.push_back(static_cast<int64_t>(std::stoll(item)));
  }
  if (vals.size() != 4) {
    return false;
  }
  out = {vals[0], vals[1], vals[2], vals[3]};
  return true;
}

static std::string shape_to_string(const Shape &s) {
  std::ostringstream oss;
  oss << s.b << "x" << s.h << "x" << s.n << "x" << s.d;
  return oss.str();
}

static bool valid_shape(const Shape &s) {
  if (s.b <= 0 || s.h <= 0 || s.n <= 0 || s.d <= 0) {
    return false;
  }
  if ((s.n % 64) != 0) {
    return false;
  }
  if (s.d != 64 && s.d != 128) {
    return false;
  }
  return true;
}

static torch::Tensor sdpa_ref(const torch::Tensor &q, const torch::Tensor &k,
                              const torch::Tensor &v) {
  return at::scaled_dot_product_attention(
      q, k, v,
      /*attn_mask=*/c10::nullopt,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*scale=*/c10::nullopt,
      /*enable_gqa=*/false);
}

static double effective_tflops(const Shape &s, double avg_ms) {
  // Approximate only matmul FLOPs: QK^T + PV = 4 * B * H * N * N * D
  const double flops = 4.0 * static_cast<double>(s.b) *
                       static_cast<double>(s.h) * static_cast<double>(s.n) *
                       static_cast<double>(s.n) * static_cast<double>(s.d);
  return flops / (avg_ms * 1e-3) / 1.0e12;
}

static std::string format_num(double x, int prec = 4) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(prec) << x;
  return oss.str();
}

static std::string now_local_string() {
  std::time_t t = std::time(nullptr);
  std::tm tm_buf{};
  localtime_r(&t, &tm_buf);
  char buf[64] = {};
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", &tm_buf);
  return std::string(buf);
}

int main(int argc, char **argv) {
  Args args;
  int64_t b = -1, h = -1, n = -1, d = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--shape" && i + 1 < argc) {
      Shape s{};
      if (!parse_shape(argv[++i], s)) {
        std::cerr << "Invalid --shape: " << argv[i] << "\n";
        return 1;
      }
      args.shapes.push_back(s);
    } else if (arg == "--b" && i + 1 < argc) {
      b = std::stoll(argv[++i]);
    } else if (arg == "--h" && i + 1 < argc) {
      h = std::stoll(argv[++i]);
    } else if (arg == "--n" && i + 1 < argc) {
      n = std::stoll(argv[++i]);
    } else if (arg == "--d" && i + 1 < argc) {
      d = std::stoll(argv[++i]);
    } else if (arg == "--ref" && i + 1 < argc) {
      args.ref = argv[++i];
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

  if (b > 0 || h > 0 || n > 0 || d > 0) {
    if (b <= 0 || h <= 0 || n <= 0 || d <= 0) {
      std::cerr << "If using --b/--h/--n/--d, all must be > 0\n";
      return 1;
    }
    args.shapes.push_back({b, h, n, d});
  }

  if (args.ref != "libtorch_sdpa" && args.ref != "none") {
    std::cerr << "Unsupported --ref: " << args.ref
              << " (expected libtorch_sdpa or none)\n";
    return 1;
  }

  if (args.shapes.empty()) {
    args.shapes.push_back({8, 16, 1024, 64});
    args.shapes.push_back({8, 16, 2048, 64});
    args.shapes.push_back({8, 16, 1024, 128});
    args.shapes.push_back({8, 16, 2048, 128});
  }

  for (const auto &shape : args.shapes) {
    if (!valid_shape(shape)) {
      std::cerr << "Unsupported shape " << shape_to_string(shape)
                << ": require B/H/N/D > 0, N % 64 == 0, D in {64, 128}\n";
      return 1;
    }
  }

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available\n";
    return 1;
  }

  torch::NoGradGuard no_grad;
  auto device = torch::Device(torch::kCUDA, 0);
  auto opt = torch::TensorOptions().device(device).dtype(torch::kFloat16);

  at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t stream = current_stream.stream();

  cudaDeviceProp prop{};
  sion::bench::cuda_check(cudaGetDeviceProperties(&prop, 0),
                          "cuda get device properties failed");
  std::string device_name = prop.name;

  std::ofstream out(args.out, std::ios::trunc);
  out << "# FlashAttention Benchmark Compare (ref: " << args.ref << ")\n\n";
  out << "- Generated: " << now_local_string() << "\n";
  out << "- kernel: " << args.kernel << "\n";
  out << "- device: " << device_name << "\n";
  out << "- dtype: f16\n";
  out << "- config: warmup=" << args.bench_cfg.warmup
      << ", repeat=" << args.bench_cfg.repeat
      << ", iters=" << args.bench_cfg.iters << "\n";
  out << "- note: `cuBLAS*` columns below are kept for compatibility with "
         "existing plot scripts; actual ref backend is `" << args.ref
      << "`.\n\n";
  out << "| Shape | Sion avg_ms | cuBLAS avg_ms (ref) | Sion speedup vs cuBLAS "
         "| Sion TFLOPS | cuBLAS TFLOPS | Sion TFLOPS ratio | Winner |\n";
  out << "|-------|-------------|---------------------|------------------------|"
         "-------------|---------------|-------------------|--------|\n";

  for (const auto &shape : args.shapes) {
    auto q = torch::randn({shape.b, shape.h, shape.n, shape.d}, opt);
    auto k = torch::randn({shape.b, shape.h, shape.n, shape.d}, opt);
    auto v = torch::randn({shape.b, shape.h, shape.n, shape.d}, opt);
    auto o = torch::empty_like(q);
    torch::Tensor ref_out;

    half *dQ = reinterpret_cast<half *>(q.data_ptr<at::Half>());
    half *dK = reinterpret_cast<half *>(k.data_ptr<at::Half>());
    half *dV = reinterpret_cast<half *>(v.data_ptr<at::Half>());
    half *dO = reinterpret_cast<half *>(o.data_ptr<at::Half>());

    auto felix_launch = [&](cudaStream_t s) {
      felix::FelixStatus status;
      if (shape.d == 64) {
        status = felix::ampere_flash_attn_launch<64, 64>(
            dQ, dK, dV, dO, static_cast<uint32_t>(shape.h),
            static_cast<uint32_t>(shape.b), static_cast<uint32_t>(shape.n), s,
            args.kernel);
      } else if (shape.d == 128) {
        status = felix::ampere_flash_attn_launch<128, 64>(
            dQ, dK, dV, dO, static_cast<uint32_t>(shape.h),
            static_cast<uint32_t>(shape.b), static_cast<uint32_t>(shape.n), s,
            args.kernel);
      } else {
        std::cerr << "Unsupported head dimension in launch: " << shape.d
                  << "\n";
        std::exit(1);
      }
      if (!status.ok()) {
        std::cerr << "Sion flash attention launch failed: " << status.str()
                  << "\n";
        std::exit(1);
      }
    };

    auto felix_stats =
        sion::bench::run_kernel_bench(felix_launch, args.bench_cfg, stream);
    felix_stats.tflops = effective_tflops(shape, felix_stats.avg_ms);

    double ref_avg_ms = 0.0;
    double ref_eff_tflops = 0.0;
    std::string winner = "Sion";
    if (args.ref == "libtorch_sdpa") {
      auto ref_launch = [&](cudaStream_t) { ref_out = sdpa_ref(q, k, v); };
      auto ref_stats =
          sion::bench::run_kernel_bench(ref_launch, args.bench_cfg, stream);
      ref_avg_ms = ref_stats.avg_ms;
      ref_eff_tflops = effective_tflops(shape, ref_avg_ms);
      winner = (felix_stats.avg_ms <= ref_avg_ms) ? "Sion" : "Ref";
    } else {
      winner = "Sion";
    }

    const double speedup =
        (ref_avg_ms > 0.0) ? (ref_avg_ms / felix_stats.avg_ms) : 0.0;
    const double tflops_ratio =
        (ref_eff_tflops > 0.0) ? (felix_stats.tflops / ref_eff_tflops) : 0.0;

    std::cout << "[Sion] fa " << shape_to_string(shape)
              << " avg_ms=" << felix_stats.avg_ms;
    if (args.ref == "libtorch_sdpa") {
      std::cout << " ref_avg_ms=" << ref_avg_ms
                << " speedup=" << speedup << "x";
    }
    std::cout << "\n";

    out << "| " << shape_to_string(shape) << " | " << format_num(felix_stats.avg_ms)
        << " | ";
    if (args.ref == "libtorch_sdpa") {
      out << format_num(ref_avg_ms);
    } else {
      out << "N/A";
    }
    out << " | ";
    if (args.ref == "libtorch_sdpa") {
      out << format_num(speedup) << "x";
    } else {
      out << "N/A";
    }
    out << " | " << format_num(felix_stats.tflops) << " | ";
    if (args.ref == "libtorch_sdpa") {
      out << format_num(ref_eff_tflops);
    } else {
      out << "N/A";
    }
    out << " | ";
    if (args.ref == "libtorch_sdpa") {
      out << format_num(tflops_ratio) << "x";
    } else {
      out << "N/A";
    }
    out << " | " << winner << " |\n";
  }

  std::cout << "[Sion] benchmarks finished, results in " << args.out << "\n";
  return 0;
}
