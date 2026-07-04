#include <felix/felix.hpp>
#include <felix/registry.hpp>

#include <sstream>

namespace felix {
namespace {

struct GemmDispatchKey {
  KernelType type;
  KernelLayout layout;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  float alpha;
  float beta;
  int cc;
};

struct FlashAttnDispatchKey {
  uint32_t head_dim;
  uint32_t block_k;
  uint32_t seq_len;
  int cc;
};

const char *kernel_type_name(KernelType type) {
  switch (type) {
  case KernelType::SGEMM:
    return "SGEMM";
  case KernelType::HGEMM:
    return "HGEMM";
  case KernelType::FlashAttention:
    return "FlashAttention";
  case KernelType::TopK:
    return "TopK";
  }
  return "Unknown";
}

const char *kernel_layout_name(KernelLayout layout) {
  switch (layout) {
  case KernelLayout::NN:
    return "NN";
  case KernelLayout::NT:
    return "NT";
  }
  return "Unknown";
}

FelixStatus current_compute_capability(int &cc) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, err,
                             "cudaGetDevice failed during Felix dispatch");
  }

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, err,
        "cudaGetDeviceProperties failed during Felix dispatch");
  }

  cc = prop.major * 10 + prop.minor;
  return {};
}

bool supports_arch(const KernelCommonMetadata &metadata, int cc) {
  if (metadata.min_cc != 0 && cc < metadata.min_cc) {
    return false;
  }
  if (metadata.max_cc != 0 && cc > metadata.max_cc) {
    return false;
  }
  return true;
}

int kernel_score(const KernelEntry &entry) {
  return entry.common.priority + (entry.stable ? 100000 : 0);
}

template <typename SupportsFn>
const KernelEntry *select_best_kernel(KernelType type, int cc,
                                      SupportsFn supports) {
  const KernelEntry *best = nullptr;
  int best_score = 0;

  for (const auto &entry : global_registry().all()) {
    if (entry.type != type || !supports_arch(entry.common, cc) ||
        !supports(entry)) {
      continue;
    }

    const int score = kernel_score(entry);
    if (best == nullptr || score > best_score) {
      best = &entry;
      best_score = score;
    }
  }

  return best;
}

bool supports_gemm(const GemmKernelMetadata &metadata,
                   const GemmDispatchKey &key) {
  if (metadata.layout != key.layout) {
    return false;
  }
  if ((key.m % metadata.align_m) != 0 || (key.n % metadata.align_n) != 0 ||
      (key.k % metadata.align_k) != 0) {
    return false;
  }
  if (metadata.requires_alpha_one_beta_zero &&
      (key.alpha != 1.0f || key.beta != 0.0f)) {
    return false;
  }
  return true;
}

const KernelEntry *select_gemm_kernel(const GemmDispatchKey &key) {
  return select_best_kernel(key.type, key.cc, [&](const KernelEntry &entry) {
    auto *metadata = std::get_if<GemmKernelMetadata>(&entry.metadata);
    return metadata != nullptr && supports_gemm(*metadata, key);
  });
}

bool supports_flash_attn(const FlashAttnKernelMetadata &metadata,
                         const FlashAttnDispatchKey &key) {
  if (metadata.head_dim != 0 && metadata.head_dim != key.head_dim) {
    return false;
  }
  if (metadata.block_k != 0 && metadata.block_k != key.block_k) {
    return false;
  }
  if (metadata.seq_len_multiple != 0 &&
      (key.seq_len % metadata.seq_len_multiple) != 0) {
    return false;
  }
  return true;
}

const KernelEntry *select_flash_attn_kernel(const FlashAttnDispatchKey &key) {
  return select_best_kernel(
      KernelType::FlashAttention, key.cc, [&](const KernelEntry &entry) {
        auto *metadata = std::get_if<FlashAttnKernelMetadata>(&entry.metadata);
        return metadata != nullptr && supports_flash_attn(*metadata, key);
      });
}

const KernelEntry *find_kernel_by_name(const std::string &name) {
  for (const auto &entry : global_registry().all()) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

FelixStatus no_matching_gemm_kernel(const GemmDispatchKey &key) {
  std::ostringstream oss;
  oss << "No matching Felix GEMM kernel found"
      << " type=" << kernel_type_name(key.type)
      << " layout=" << kernel_layout_name(key.layout) << " cc=" << key.cc
      << " M=" << key.m << " N=" << key.n << " K=" << key.k
      << " alpha=" << key.alpha << " beta=" << key.beta;
  return FelixStatus::make(FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
                           oss.str());
}

FelixStatus no_matching_flash_attn_kernel(const FlashAttnDispatchKey &key) {
  std::ostringstream oss;
  oss << "No matching Felix FlashAttention kernel found"
      << " cc=" << key.cc << " head_dim=" << key.head_dim
      << " block_k=" << key.block_k << " seq_len=" << key.seq_len;
  return FelixStatus::make(FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
                           oss.str());
}

FelixStatus wrong_named_kernel_type(const KernelEntry &entry,
                                    KernelType expected,
                                    const std::string &name) {
  std::ostringstream oss;
  oss << "Named kernel '" << name << "' is " << kernel_type_name(entry.type)
      << ", expected " << kernel_type_name(expected);
  return FelixStatus::make(FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
                           oss.str());
}

FelixStatus missing_named_kernel(const std::string &kind,
                                 const std::string &name) {
  return FelixStatus::make(FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
                           "No matching named " + kind + " kernel: " + name);
}

const KernelEntry *checked_named_kernel(const std::string &name,
                                        KernelType expected,
                                        FelixStatus &status) {
  auto *entry = find_kernel_by_name(name);
  if (entry == nullptr) {
    status = missing_named_kernel(kernel_type_name(expected), name);
    return nullptr;
  }
  if (entry->type != expected) {
    status = wrong_named_kernel_type(*entry, expected, name);
    return nullptr;
  }
  return entry;
}

FelixStatus named_kernel_not_supported(const KernelEntry &entry,
                                       const std::string &reason) {
  return FelixStatus::make(
      FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
      "Named kernel '" + entry.name +
          "' does not support this dispatch request: " + reason);
}

const KernelEntry *checked_named_gemm_kernel(const std::string &name,
                                             const GemmDispatchKey &key,
                                             FelixStatus &status) {
  auto *entry = checked_named_kernel(name, key.type, status);
  if (entry == nullptr) {
    return nullptr;
  }
  if (!supports_arch(entry->common, key.cc)) {
    status = named_kernel_not_supported(*entry, "compute capability cc=" +
                                                    std::to_string(key.cc));
    return nullptr;
  }
  auto *metadata = std::get_if<GemmKernelMetadata>(&entry->metadata);
  if (metadata == nullptr || !supports_gemm(*metadata, key)) {
    std::ostringstream oss;
    oss << "type=" << kernel_type_name(key.type)
        << " layout=" << kernel_layout_name(key.layout) << " M=" << key.m
        << " N=" << key.n << " K=" << key.k << " alpha=" << key.alpha
        << " beta=" << key.beta;
    status = named_kernel_not_supported(*entry, oss.str());
    return nullptr;
  }
  return entry;
}

const KernelEntry *
checked_named_flash_attn_kernel(const std::string &name,
                                const FlashAttnDispatchKey &key,
                                FelixStatus &status) {
  auto *entry = checked_named_kernel(name, KernelType::FlashAttention, status);
  if (entry == nullptr) {
    return nullptr;
  }
  if (!supports_arch(entry->common, key.cc)) {
    status = named_kernel_not_supported(*entry, "compute capability cc=" +
                                                    std::to_string(key.cc));
    return nullptr;
  }
  auto *metadata = std::get_if<FlashAttnKernelMetadata>(&entry->metadata);
  if (metadata == nullptr || !supports_flash_attn(*metadata, key)) {
    std::ostringstream oss;
    oss << "head_dim=" << key.head_dim << " block_k=" << key.block_k
        << " seq_len=" << key.seq_len;
    status = named_kernel_not_supported(*entry, oss.str());
    return nullptr;
  }
  return entry;
}

const KernelEntry *checked_named_topk_kernel(const std::string &name,
                                             uint32_t k, bool largest, int cc,
                                             FelixStatus &status) {
  auto *entry = checked_named_kernel(name, KernelType::TopK, status);
  if (entry == nullptr) {
    return nullptr;
  }
  if (!supports_arch(entry->common, cc)) {
    status = named_kernel_not_supported(*entry, "compute capability cc=" +
                                                    std::to_string(cc));
    return nullptr;
  }
  auto *metadata = std::get_if<TopKKernelMetadata>(&entry->metadata);
  const bool topk_supported =
      metadata != nullptr && (metadata->max_k == 0 || k <= metadata->max_k) &&
      (largest ? metadata->supports_largest : metadata->supports_smallest);
  if (!topk_supported) {
    std::ostringstream oss;
    oss << "k=" << k << " largest=" << largest;
    status = named_kernel_not_supported(*entry, oss.str());
    return nullptr;
  }
  return entry;
}

FelixStatus dispatch_sgemm_entry(const KernelEntry &entry, uint32_t M,
                                 uint32_t N, uint32_t K, float alpha,
                                 float const *A, float const *B, float beta,
                                 float *C, cudaStream_t stream) {
  auto fn = std::get_if<SgemmKernelFn>(&entry.fn);
  if (fn == nullptr) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "Selected kernel is not an SGEMM launcher");
  }
  return (*fn)(M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus dispatch_hgemm_entry(const KernelEntry &entry, uint32_t M,
                                 uint32_t N, uint32_t K, float alpha,
                                 half const *A, half const *B, float beta,
                                 half *C, cudaStream_t stream) {
  auto fn = std::get_if<HgemmKernelFn>(&entry.fn);
  if (fn == nullptr) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "Selected kernel is not an HGEMM launcher");
  }
  return (*fn)(M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus dispatch_flash_attn_entry(const KernelEntry &entry, half *Q,
                                      half *K, half *V, half *O, uint32_t heads,
                                      uint32_t batch_size, uint32_t QKV_seqlen,
                                      cudaStream_t stream) {
  auto fn = std::get_if<FlashAttnKernelFn>(&entry.fn);
  if (fn == nullptr) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "Selected kernel is not a FlashAttention launcher");
  }
  return (*fn)(Q, K, V, O, heads, batch_size, QKV_seqlen, stream);
}

FelixStatus dispatch_topk_entry(const KernelEntry &entry, float const *data,
                                float *out, uint32_t num_slices,
                                uint32_t slice_size, uint32_t k, bool largest,
                                cudaStream_t stream) {
  auto fn = std::get_if<TopKKernelFn>(&entry.fn);
  if (fn == nullptr) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "Selected kernel is not a TopK launcher");
  }
  return (*fn)(data, out, num_slices, slice_size, k, largest, stream);
}

} // namespace

FelixStatus sgemm_f32_launch(uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float const *A, float const *B, float beta,
                             float *C, cudaStream_t stream) {
  int cc = 0;
  auto status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const GemmDispatchKey key{
      KernelType::SGEMM, KernelLayout::NN, M, N, K, alpha, beta, cc};
  auto *entry = select_gemm_kernel(key);
  if (entry == nullptr) {
    return no_matching_gemm_kernel(key);
  }
  return dispatch_sgemm_entry(*entry, M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus sgemm_f32_launch_by_name(uint32_t M, uint32_t N, uint32_t K,
                                     float alpha, float const *A,
                                     float const *B, float beta, float *C,
                                     cudaStream_t stream,
                                     const std::string &kernel_name) {
  int cc = 0;
  FelixStatus status;
  status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const GemmDispatchKey key{
      KernelType::SGEMM, KernelLayout::NN, M, N, K, alpha, beta, cc};
  auto *entry = checked_named_gemm_kernel(kernel_name, key, status);
  if (entry == nullptr) {
    return status;
  }
  return dispatch_sgemm_entry(*entry, M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus hgemm_f16_launch(uint32_t M, uint32_t N, uint32_t K, float alpha,
                             half const *A, half const *B, float beta, half *C,
                             cudaStream_t stream) {
  int cc = 0;
  auto status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const GemmDispatchKey key{
      KernelType::HGEMM, KernelLayout::NN, M, N, K, alpha, beta, cc};
  auto *entry = select_gemm_kernel(key);
  if (entry == nullptr) {
    return no_matching_gemm_kernel(key);
  }
  return dispatch_hgemm_entry(*entry, M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus hgemm_f16_nt_launch(uint32_t M, uint32_t N, uint32_t K, float alpha,
                                half const *A, half const *B, float beta,
                                half *C, cudaStream_t stream) {
  int cc = 0;
  auto status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const GemmDispatchKey key{
      KernelType::HGEMM, KernelLayout::NT, M, N, K, alpha, beta, cc};
  auto *entry = select_gemm_kernel(key);
  if (entry == nullptr) {
    return no_matching_gemm_kernel(key);
  }
  return dispatch_hgemm_entry(*entry, M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus hgemm_f16_launch_by_name(uint32_t M, uint32_t N, uint32_t K,
                                     float alpha, half const *A, half const *B,
                                     float beta, half *C, cudaStream_t stream,
                                     const std::string &kernel_name) {
  int cc = 0;
  FelixStatus status;
  status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const GemmDispatchKey key{
      KernelType::HGEMM, KernelLayout::NN, M, N, K, alpha, beta, cc};
  auto *entry = checked_named_gemm_kernel(kernel_name, key, status);
  if (entry == nullptr) {
    return status;
  }
  return dispatch_hgemm_entry(*entry, M, N, K, alpha, A, B, beta, C, stream);
}

FelixStatus topk_f32_radix_select_launch(float const *data, float *out,
                                         uint32_t num_slices,
                                         uint32_t slice_size, uint32_t k,
                                         bool largest, cudaStream_t stream,
                                         const std::string &kernel_name) {
  int cc = 0;
  FelixStatus status;
  status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  auto *entry = checked_named_topk_kernel(kernel_name, k, largest, cc, status);
  if (entry == nullptr) {
    return status;
  }
  return dispatch_topk_entry(*entry, data, out, num_slices, slice_size, k,
                             largest, stream);
}

template <>
FelixStatus flash_attn_f16_launch<64, 64>(half *Q, half *K, half *V, half *O,
                                          uint32_t heads, uint32_t batch_size,
                                          uint32_t QKV_seqlen,
                                          cudaStream_t stream) {
  int cc = 0;
  auto status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const FlashAttnDispatchKey key{64, 64, QKV_seqlen, cc};
  auto *entry = select_flash_attn_kernel(key);
  if (entry == nullptr) {
    return no_matching_flash_attn_kernel(key);
  }
  return dispatch_flash_attn_entry(*entry, Q, K, V, O, heads, batch_size,
                                   QKV_seqlen, stream);
}

template <>
FelixStatus flash_attn_f16_launch<128, 64>(half *Q, half *K, half *V, half *O,
                                           uint32_t heads, uint32_t batch_size,
                                           uint32_t QKV_seqlen,
                                           cudaStream_t stream) {
  int cc = 0;
  auto status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const FlashAttnDispatchKey key{128, 64, QKV_seqlen, cc};
  auto *entry = select_flash_attn_kernel(key);
  if (entry == nullptr) {
    return no_matching_flash_attn_kernel(key);
  }
  return dispatch_flash_attn_entry(*entry, Q, K, V, O, heads, batch_size,
                                   QKV_seqlen, stream);
}

template <>
FelixStatus flash_attn_f16_launch_by_name<64, 64>(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t QKV_seqlen, cudaStream_t stream, const std::string &kernel_name) {
  int cc = 0;
  FelixStatus status;
  status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const FlashAttnDispatchKey key{64, 64, QKV_seqlen, cc};
  auto *entry = checked_named_flash_attn_kernel(kernel_name, key, status);
  if (entry == nullptr) {
    return status;
  }
  return dispatch_flash_attn_entry(*entry, Q, K, V, O, heads, batch_size,
                                   QKV_seqlen, stream);
}

template <>
FelixStatus flash_attn_f16_launch_by_name<128, 64>(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t QKV_seqlen, cudaStream_t stream, const std::string &kernel_name) {
  int cc = 0;
  FelixStatus status;
  status = current_compute_capability(cc);
  if (!status.ok()) {
    return status;
  }

  const FlashAttnDispatchKey key{128, 64, QKV_seqlen, cc};
  auto *entry = checked_named_flash_attn_kernel(kernel_name, key, status);
  if (entry == nullptr) {
    return status;
  }
  return dispatch_flash_attn_entry(*entry, Q, K, V, O, heads, batch_size,
                                   QKV_seqlen, stream);
}

} // namespace felix
