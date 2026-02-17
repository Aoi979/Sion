#include <felix/felix.hpp>
#include <felix/registry.hpp>
#include <stdexcept>

namespace felix {

template <typename... Args>
FelixStatus dispatch_kernel(KernelType type, const std::string &name,
                            Args... args) {
  static_assert((!std::is_reference_v<Args> && ...),
                "dispatch_kernel does not allow reference arguments");

  for (auto &k : global_registry().all()) {
    if (k.type == type && k.name == name) {
      auto fn = reinterpret_cast<FelixStatus (*)(Args...)>(k.fn_ptr);
      return fn(std::forward<Args>(args)...);
    }
  }
  throw std::runtime_error("No matching kernel found");
}

FelixStatus ampere_sgemm_launch(uint32_t M, uint32_t N, uint32_t K, float alpha,
                                float const *A, float const *B, float beta,
                                float *C, cudaStream_t stream,
                                const std::string &kernel_name) {
  return felix::dispatch_kernel(felix::KernelType::SGEMM, kernel_name, M, N, K,
                                alpha, A, B, beta, C, stream);
}

template <>
FelixStatus ampere_flash_attn_launch<64, 64>(half *Q, half *K, half *V, half *O,
                                             uint32_t heads,
                                             uint32_t batch_size,
                                             uint32_t QKV_seqlen,
                                             cudaStream_t stream,
                                             const std::string &kernel_name) {
  return felix::dispatch_kernel(felix::KernelType::FlashAttn_64, kernel_name, Q,
                                K, V, O, heads, batch_size, QKV_seqlen, stream);
}

template <>
FelixStatus ampere_flash_attn_launch<128, 64>(half *Q, half *K, half *V,
                                              half *O, uint32_t heads,
                                              uint32_t batch_size,
                                              uint32_t QKV_seqlen,
                                              cudaStream_t stream,
                                              const std::string &kernel_name) {
  return felix::dispatch_kernel(felix::KernelType::FlashAttn_128, kernel_name,
                                Q, K, V, O, heads, batch_size, QKV_seqlen,
                                stream);
}

} // namespace felix
