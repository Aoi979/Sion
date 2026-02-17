#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace felix {

enum class KernelType : uint8_t { HGEMM, SGEMM, FlashAttn_64, FlashAttn_128 };

struct KernelEntry {
  KernelType type;
  std::string name;
  void *fn_ptr;
  bool stable;
};

class KernelRegistry {
public:
  void add(KernelEntry k);
  const std::vector<KernelEntry> &all() const;

private:
  std::vector<KernelEntry> entries;
};

KernelRegistry &global_registry();

} // namespace felix

#define REGISTER_KERNEL(name, entry) \
static bool _##name = [](){ \
    felix::global_registry().add(entry); \
    return true; \
}()