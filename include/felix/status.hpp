#pragma once
#include <cuda_runtime.h>
#include <source_location>
#include <string>
namespace felix {
struct FelixStatus {

  enum class Type {
    SUCCESS,
    KERNEL_LAUNCH_FAILED,
    KERNEL_RUNTIME_ERROR,
    API_ERROR
  };

  Type type;
  cudaError_t cuda_code;
  std::string message;
  std::string file;
  int line;
  std::string func;

  FelixStatus();

  FelixStatus(Type t, cudaError_t code, std::string msg, std::string file,
              int line, std::string func);

  [[nodiscard]]
  bool ok() const;

  [[nodiscard]]
  std::string str() const;

  static FelixStatus
  make(Type t, cudaError_t code, std::string msg = "",
       const std::source_location &loc = std::source_location::current());

  static FelixStatus success() noexcept { return FelixStatus(); }

private:
  const char *type_string() const;
};
} // namespace felix
