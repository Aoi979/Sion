#include <felix/registry.hpp>
#include <utility>

namespace felix {

void KernelRegistry::add(KernelEntry k) { entries.push_back(std::move(k)); }

const std::vector<KernelEntry> &KernelRegistry::all() const { return entries; }

KernelRegistry &global_registry() {
  static KernelRegistry instance;
  return instance;
}

} // namespace felix
