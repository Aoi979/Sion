#include <felix/registry.hpp>
#include <utility>

namespace felix {

void KernelRegistry::add(KernelEntry k) { entries.push_back(std::move(k)); }

const std::vector<KernelEntry> &KernelRegistry::all() const { return entries; }

KernelRegistry &global_registry() {
  static KernelRegistry instance;
  return instance;
}

KernelEntry make_sgemm_kernel(std::string name, SgemmKernelFn fn, bool stable,
                              KernelCommonMetadata common,
                              GemmKernelMetadata metadata) {
  return KernelEntry{KernelType::SGEMM, std::move(name), fn, stable, common,
                     metadata};
}

KernelEntry make_hgemm_kernel(std::string name, HgemmKernelFn fn, bool stable,
                              KernelCommonMetadata common,
                              GemmKernelMetadata metadata) {
  return KernelEntry{KernelType::HGEMM, std::move(name), fn, stable, common,
                     metadata};
}

KernelEntry make_topk_kernel(std::string name, TopKKernelFn fn, bool stable,
                             KernelCommonMetadata common,
                             TopKKernelMetadata metadata) {
  return KernelEntry{KernelType::TopK, std::move(name), fn,
                     stable,           common,          metadata};
}

KernelEntry make_flash_attn_kernel(std::string name, FlashAttnKernelFn fn,
                                   bool stable, KernelCommonMetadata common,
                                   FlashAttnKernelMetadata metadata) {
  return KernelEntry{KernelType::FlashAttention,
                     std::move(name),
                     fn,
                     stable,
                     common,
                     metadata};
}

} // namespace felix
