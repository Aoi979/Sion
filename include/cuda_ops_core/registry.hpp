#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_ops_core/status.hpp>
#include <string>
#include <variant>
#include <vector>

namespace cuda_ops_core {

enum class KernelType : uint8_t { SGEMM, HGEMM, FlashAttention, TopK };

enum class KernelLayout : uint8_t { NN, NT };

using SgemmKernelFn = Status (*)(uint32_t, uint32_t, uint32_t, float,
                                      float const *, float const *, float,
                                      float *, cudaStream_t);
using HgemmKernelFn = Status (*)(uint32_t, uint32_t, uint32_t, float,
                                      half const *, half const *, float, half *,
                                      cudaStream_t);
using TopKKernelFn = Status (*)(float const *, float *, uint32_t, uint32_t,
                                     uint32_t, bool, cudaStream_t);
using FlashAttnKernelFn = Status (*)(half *, half *, half *, half *,
                                          uint32_t, uint32_t, uint32_t,
                                          cudaStream_t);
using KernelFunction =
    std::variant<SgemmKernelFn, HgemmKernelFn, TopKKernelFn, FlashAttnKernelFn>;

struct KernelCommonMetadata {
  int min_cc = 0;
  int max_cc = 0;
  int priority = 0;
  uint32_t required_dynamic_smem_bytes = 0;
  uint32_t required_threads_per_block = 0;
};

struct GemmKernelMetadata {
  KernelLayout layout = KernelLayout::NN;
  uint32_t align_m = 1;
  uint32_t align_n = 1;
  uint32_t align_k = 1;
  bool requires_alpha_one_beta_zero = false;
};

struct FlashAttnKernelMetadata {
  uint32_t head_dim = 0;
  uint32_t block_q = 1;
  uint32_t block_k = 1;
  uint32_t seq_len_multiple = 1;
};

struct TopKKernelMetadata {
  uint32_t max_k = 0;
  bool supports_largest = true;
  bool supports_smallest = true;
};

using KernelMetadata = std::variant<GemmKernelMetadata, FlashAttnKernelMetadata,
                                    TopKKernelMetadata>;

struct KernelEntry {
  KernelType type;
  std::string name;
  KernelFunction fn;
  bool stable;
  KernelCommonMetadata common;
  KernelMetadata metadata;
};

class KernelRegistry {
public:
  void add(KernelEntry k);
  const std::vector<KernelEntry> &all() const;

private:
  std::vector<KernelEntry> entries;
};

KernelRegistry &global_registry();

KernelEntry make_sgemm_kernel(std::string name, SgemmKernelFn fn, bool stable,
                              KernelCommonMetadata common = {},
                              GemmKernelMetadata metadata = {});
KernelEntry make_hgemm_kernel(std::string name, HgemmKernelFn fn, bool stable,
                              KernelCommonMetadata common = {},
                              GemmKernelMetadata metadata = {});
KernelEntry make_topk_kernel(std::string name, TopKKernelFn fn, bool stable,
                             KernelCommonMetadata common = {},
                             TopKKernelMetadata metadata = {});
KernelEntry make_flash_attn_kernel(std::string name, FlashAttnKernelFn fn,
                                   bool stable,
                                   KernelCommonMetadata common = {},
                                   FlashAttnKernelMetadata metadata = {});

} // namespace cuda_ops_core

#define REGISTER_KERNEL(name, entry)                                           \
  static bool _##name = []() {                                                 \
    cuda_ops_core::global_registry().add(entry);                                       \
    return true;                                                               \
  }()
