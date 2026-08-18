#include "../core/cuda_utils.hpp"
#include "../core/json.hpp"
#include "../core/timer.hpp"

#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/flash_attention/kernels/sm80_flash_attn_f16_mma16816_v2.cuh>
#include <cuda_ops_core/registry.hpp>

#include <cuda_fp16.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct Args {
  std::string op;
  std::string layer = "cuda_ops_core";
  std::string kernel = "auto";
  std::string shape;
  std::string out;
  cuda_ops::bench::TimingConfig timing;
  bool list_kernels = false;
};

[[noreturn]] void usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog
      << " --op sgemm|hgemm|flash_attention --layer cuda_ops_core|raw --shape SHAPE "
         "[--kernel NAME|auto]\n"
      << "       [--warmup N] [--repeat N] [--iters N] [--min-sample-ms MS] "
         "[--out FILE]\n"
      << "  " << prog << " --list-kernels [--op OP]\n\n"
      << "Shapes:\n"
      << "  GEMM: MxNxK\n"
      << "  FlashAttention: BxHxNxD\n";
  std::exit(2);
}

std::uint32_t parse_u32(std::string_view text, const char *name) {
  std::uint32_t value = 0;
  auto *first = text.data();
  auto *last = text.data() + text.size();
  auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{} || result.ptr != last) {
    throw std::runtime_error(std::string("invalid ") + name + ": " +
                             std::string(text));
  }
  return value;
}

std::vector<std::uint32_t> parse_shape_parts(const std::string &shape) {
  std::vector<std::uint32_t> parts;
  std::size_t pos = 0;
  while (pos <= shape.size()) {
    const auto next = shape.find('x', pos);
    const auto end = next == std::string::npos ? shape.size() : next;
    parts.push_back(parse_u32(std::string_view(shape).substr(pos, end - pos),
                              "shape component"));
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return parts;
}

struct GemmShape {
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
};

struct FlashShape {
  std::uint32_t b = 0;
  std::uint32_t h = 0;
  std::uint32_t n = 0;
  std::uint32_t d = 0;
};

GemmShape parse_gemm_shape(const std::string &shape) {
  const auto parts = parse_shape_parts(shape);
  if (parts.size() != 3) {
    throw std::runtime_error("GEMM shape must be MxNxK");
  }
  return {parts[0], parts[1], parts[2]};
}

FlashShape parse_flash_shape(const std::string &shape) {
  const auto parts = parse_shape_parts(shape);
  if (parts.size() != 4) {
    throw std::runtime_error("FlashAttention shape must be BxHxNxD");
  }
  return {parts[0], parts[1], parts[2], parts[3]};
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else if (arg == "--list-kernels") {
      args.list_kernels = true;
    } else if (arg == "--op") {
      args.op = need_value("--op");
    } else if (arg == "--layer") {
      args.layer = need_value("--layer");
    } else if (arg == "--kernel") {
      args.kernel = need_value("--kernel");
    } else if (arg == "--shape") {
      args.shape = need_value("--shape");
    } else if (arg == "--warmup") {
      args.timing.warmup =
          static_cast<int>(parse_u32(need_value("--warmup"), "warmup"));
    } else if (arg == "--repeat") {
      args.timing.repeat =
          static_cast<int>(parse_u32(need_value("--repeat"), "repeat"));
    } else if (arg == "--iters") {
      args.timing.iters =
          static_cast<int>(parse_u32(need_value("--iters"), "iters"));
    } else if (arg == "--max-iters") {
      args.timing.max_iters =
          static_cast<int>(parse_u32(need_value("--max-iters"), "max-iters"));
    } else if (arg == "--min-sample-ms") {
      args.timing.min_sample_ms = std::stod(need_value("--min-sample-ms"));
    } else if (arg == "--out") {
      args.out = need_value("--out");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (!args.list_kernels &&
      (args.op.empty() || args.layer.empty() || args.shape.empty())) {
    usage(argv[0]);
  }
  return args;
}

const char *kernel_type_name(cuda_ops_core::KernelType type) {
  switch (type) {
  case cuda_ops_core::KernelType::SGEMM:
    return "sgemm";
  case cuda_ops_core::KernelType::HGEMM:
    return "hgemm";
  case cuda_ops_core::KernelType::FlashAttention:
    return "flash_attention";
  case cuda_ops_core::KernelType::TopK:
    return "topk";
  }
  return "unknown";
}

bool matches_op(cuda_ops_core::KernelType type, const std::string &op) {
  return op.empty() || op == kernel_type_name(type);
}

void list_kernels(const std::string &op) {
  std::cout << "[\n";
  bool first = true;
  for (const auto &entry : cuda_ops_core::global_registry().all()) {
    if (!matches_op(entry.type, op)) {
      continue;
    }
    if (!first) {
      std::cout << ",\n";
    }
    first = false;
    std::cout << "  {\"op\":\"" << kernel_type_name(entry.type)
              << "\",\"name\":\"" << cuda_ops::bench::json_escape(entry.name)
              << "\",\"stable\":" << (entry.stable ? "true" : "false")
              << ",\"priority\":" << entry.common.priority << "}";
  }
  std::cout << "\n]\n";
}

void checked_status(const cuda_ops_core::Status &status) {
  if (!status.ok()) {
    throw std::runtime_error(status.str());
  }
}

cuda_ops::bench::LaunchFn make_sgemm_cuda_ops_core_launch(const GemmShape &shape,
                                              const std::string &kernel,
                                              cuda_ops::bench::DeviceBuffer<float> &a,
                                              cuda_ops::bench::DeviceBuffer<float> &b,
                                              cuda_ops::bench::DeviceBuffer<float> &c) {
  return [=, &a, &b, &c](cudaStream_t stream) {
    cuda_ops_core::Status status;
    if (kernel == "auto") {
      status = cuda_ops_core::sgemm_f32_launch(shape.m, shape.n, shape.k, 1.0f, a.get(),
                                       b.get(), 0.0f, c.get(), stream);
    } else {
      status = cuda_ops_core::sgemm_f32_launch_by_name(
          shape.m, shape.n, shape.k, 1.0f, a.get(), b.get(), 0.0f, c.get(),
          stream, kernel);
    }
    checked_status(status);
  };
}

cuda_ops::bench::LaunchFn make_hgemm_cuda_ops_core_launch(const GemmShape &shape,
                                              const std::string &kernel,
                                              cuda_ops::bench::DeviceBuffer<half> &a,
                                              cuda_ops::bench::DeviceBuffer<half> &b,
                                              cuda_ops::bench::DeviceBuffer<half> &c) {
  return [=, &a, &b, &c](cudaStream_t stream) {
    cuda_ops_core::Status status;
    if (kernel == "auto") {
      status = cuda_ops_core::hgemm_f16_launch(shape.m, shape.n, shape.k, 1.0f, a.get(),
                                       b.get(), 0.0f, c.get(), stream);
    } else {
      status = cuda_ops_core::hgemm_f16_launch_by_name(
          shape.m, shape.n, shape.k, 1.0f, a.get(), b.get(), 0.0f, c.get(),
          stream, kernel);
    }
    checked_status(status);
  };
}

template <int HeadDim>
cuda_ops::bench::LaunchFn
make_flash_raw_launch(const FlashShape &shape, const std::string &kernel,
                      cuda_ops::bench::DeviceBuffer<half> &q,
                      cuda_ops::bench::DeviceBuffer<half> &k,
                      cuda_ops::bench::DeviceBuffer<half> &v,
                      cuda_ops::bench::DeviceBuffer<half> &o) {
  using Config =
      sm80_flash_attn_v2::fav2_sm80::FlashAttnV2Sm80Config<HeadDim>;
  static_assert(Config::kSupported);

  const std::string canonical =
      HeadDim == 64 ? "sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2"
                    : "sm80_flash_attn_f16_hd128_bq128_bk64_mma16816_v2";
  if (kernel != "auto" && kernel != canonical) {
    throw std::runtime_error("raw flash_attention only supports " + canonical);
  }
  if (shape.n % Config::kBlockMValue != 0 ||
      shape.n % Config::kBlockNValue != 0) {
    throw std::runtime_error("raw flash_attention shape is not tile aligned");
  }

  auto kernel_fptr =
      sm80_flash_attn_v2::flash_attn_v2<HeadDim, Config::kBlockMValue,
                                        Config::kBlockNValue>;
  if constexpr (Config::kSmemBytes >= 48 * 1024) {
    cuda_ops::bench::cuda_check(
        cudaFuncSetAttribute(kernel_fptr,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Config::kSmemBytes),
        "cudaFuncSetAttribute(raw flash smem)");
  }

  const int heads = static_cast<int>(shape.h);
  const int batch = static_cast<int>(shape.b);
  const int seq = static_cast<int>(shape.n);
  constexpr int d = HeadDim;
  sm80_flash_attn_v2::FlashFwdParams<HeadDim> params{
      .q = q.get(),
      .k = k.get(),
      .v = v.get(),
      .o = o.get(),
      .batch_size = batch,
      .seqlen_q = seq,
      .seqlen_k = seq,
      .heads_q = heads,
      .heads_k = heads,
      .q_heads_per_kv_head = 1,
      .q_batch_stride = heads * seq * d,
      .q_row_stride = d,
      .q_head_stride = seq * d,
      .k_batch_stride = heads * seq * d,
      .k_row_stride = d,
      .k_head_stride = seq * d,
      .v_batch_stride = heads * seq * d,
      .v_row_stride = d,
      .v_head_stride = seq * d,
      .o_batch_stride = heads * seq * d,
      .o_row_stride = d,
      .o_head_stride = seq * d,
      .softmax_scale_log2 =
          1.4426950408889634074f / std::sqrt(static_cast<float>(HeadDim)),
  };

  const dim3 block(Config::kThreads);
  const dim3 grid(seq / Config::kBlockMValue, batch, heads);

  return [=](cudaStream_t stream) {
    kernel_fptr<<<grid, block, Config::kSmemBytes, stream>>>(params);
    cuda_ops::bench::cuda_check(cudaGetLastError(), "raw flash_attention launch");
  };
}

cuda_ops::bench::LaunchFn
make_flash_cuda_ops_core_launch(const FlashShape &shape, const std::string &kernel,
                        cuda_ops::bench::DeviceBuffer<half> &q,
                        cuda_ops::bench::DeviceBuffer<half> &k,
                        cuda_ops::bench::DeviceBuffer<half> &v,
                        cuda_ops::bench::DeviceBuffer<half> &o) {
  return [=, &q, &k, &v, &o](cudaStream_t stream) {
    cuda_ops_core::Status status;
    if (shape.d == 64) {
      if (kernel == "auto") {
        status = cuda_ops_core::flash_attn_f16_launch<64, 64>(
            q.get(), k.get(), v.get(), o.get(), shape.h, shape.b, shape.n,
            stream);
      } else {
        status = cuda_ops_core::flash_attn_f16_launch_by_name<64, 64>(
            q.get(), k.get(), v.get(), o.get(), shape.h, shape.b, shape.n,
            stream, kernel);
      }
    } else if (shape.d == 128) {
      if (kernel == "auto") {
        status = cuda_ops_core::flash_attn_f16_launch<128, 64>(
            q.get(), k.get(), v.get(), o.get(), shape.h, shape.b, shape.n,
            stream);
      } else {
        status = cuda_ops_core::flash_attn_f16_launch_by_name<128, 64>(
            q.get(), k.get(), v.get(), o.get(), shape.h, shape.b, shape.n,
            stream, kernel);
      }
    } else {
      throw std::runtime_error("flash_attention supports D=64 or D=128");
    }
    checked_status(status);
  };
}

struct BenchRun {
  std::string resolved_kernel;
  double work_units = 0.0;
  cuda_ops::bench::LaunchFn launch;
};

BenchRun build_run(const Args &args, cudaStream_t stream) {
  if (args.op == "sgemm") {
    if (args.layer != "cuda_ops_core") {
      throw std::runtime_error("sgemm currently supports layer=cuda_ops_core");
    }
    const auto shape = parse_gemm_shape(args.shape);
    auto a = std::make_shared<cuda_ops::bench::DeviceBuffer<float>>(
        static_cast<std::size_t>(shape.m) * shape.k);
    auto b = std::make_shared<cuda_ops::bench::DeviceBuffer<float>>(
        static_cast<std::size_t>(shape.k) * shape.n);
    auto c = std::make_shared<cuda_ops::bench::DeviceBuffer<float>>(
        static_cast<std::size_t>(shape.m) * shape.n);
    a->memset_zero(stream);
    b->memset_zero(stream);
    c->memset_zero(stream);
    cuda_ops::bench::cuda_check(cudaStreamSynchronize(stream),
                            "sgemm init sync");
    auto launch = make_sgemm_cuda_ops_core_launch(shape, args.kernel, *a, *b, *c);
    return {args.kernel,
            2.0 * static_cast<double>(shape.m) * shape.n * shape.k,
            [a, b, c, launch = std::move(launch)](cudaStream_t s) {
              launch(s);
            }};
  }

  if (args.op == "hgemm") {
    if (args.layer != "cuda_ops_core") {
      throw std::runtime_error("hgemm currently supports layer=cuda_ops_core");
    }
    const auto shape = parse_gemm_shape(args.shape);
    auto a = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(
        static_cast<std::size_t>(shape.m) * shape.k);
    auto b = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(
        static_cast<std::size_t>(shape.k) * shape.n);
    auto c = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(
        static_cast<std::size_t>(shape.m) * shape.n);
    a->memset_zero(stream);
    b->memset_zero(stream);
    c->memset_zero(stream);
    cuda_ops::bench::cuda_check(cudaStreamSynchronize(stream),
                            "hgemm init sync");
    auto launch = make_hgemm_cuda_ops_core_launch(shape, args.kernel, *a, *b, *c);
    return {args.kernel,
            2.0 * static_cast<double>(shape.m) * shape.n * shape.k,
            [a, b, c, launch = std::move(launch)](cudaStream_t s) {
              launch(s);
            }};
  }

  if (args.op == "flash_attention") {
    const auto shape = parse_flash_shape(args.shape);
    if (shape.d != 64 && shape.d != 128) {
      throw std::runtime_error("flash_attention supports D=64 or D=128");
    }
    const auto elems = static_cast<std::size_t>(shape.b) * shape.h * shape.n *
                       shape.d;
    auto q = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(elems);
    auto k = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(elems);
    auto v = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(elems);
    auto o = std::make_shared<cuda_ops::bench::DeviceBuffer<half>>(elems);
    q->memset_zero(stream);
    k->memset_zero(stream);
    v->memset_zero(stream);
    o->memset_zero(stream);
    cuda_ops::bench::cuda_check(cudaStreamSynchronize(stream),
                            "flash_attention init sync");

    cuda_ops::bench::LaunchFn launch;
    std::string resolved = args.kernel;
    if (args.layer == "cuda_ops_core") {
      launch = make_flash_cuda_ops_core_launch(shape, args.kernel, *q, *k, *v, *o);
      if (resolved == "auto") {
        resolved = shape.d == 64
                       ? "auto:sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2"
                       : "auto:sm80_flash_attn_f16_hd128_bq128_bk64_mma16816_v2";
      }
    } else if (args.layer == "raw") {
      if (shape.d == 64) {
        launch = make_flash_raw_launch<64>(shape, args.kernel, *q, *k, *v, *o);
        resolved = "sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2";
      } else {
        launch = make_flash_raw_launch<128>(shape, args.kernel, *q, *k, *v, *o);
        resolved = "sm80_flash_attn_f16_hd128_bq128_bk64_mma16816_v2";
      }
    } else {
      throw std::runtime_error("flash_attention supports layer=cuda_ops_core|raw");
    }

    const double work =
        4.0 * static_cast<double>(shape.b) * shape.h * shape.n * shape.n *
        shape.d;
    return {resolved, work,
            [q, k, v, o, launch = std::move(launch)](cudaStream_t s) {
              launch(s);
            }};
  }

  throw std::runtime_error("unsupported op: " + args.op);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.list_kernels) {
      list_kernels(args.op);
      return 0;
    }

    cudaStream_t stream{};
    cuda_ops::bench::cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    const auto device = cuda_ops::bench::current_device_info();
    auto run = build_run(args, stream);
    const auto timing = cuda_ops::bench::run_timing(run.launch, args.timing, stream);

    if (args.out.empty()) {
      cuda_ops::bench::write_result_json(std::cout, args.op, args.layer,
                                     run.resolved_kernel, args.shape, device,
                                     args.timing, timing, run.work_units);
    } else {
      std::ofstream out(args.out);
      if (!out) {
        throw std::runtime_error("failed to open output file: " + args.out);
      }
      cuda_ops::bench::write_result_json(out, args.op, args.layer,
                                     run.resolved_kernel, args.shape, device,
                                     args.timing, timing, run.work_units);
    }

    cuda_ops::bench::cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[cuda_ops_bench] " << e.what() << "\n";
    return 1;
  }
}
