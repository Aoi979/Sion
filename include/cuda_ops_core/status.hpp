#pragma once
#include <cuda_runtime.h>
#include <source_location>
#include <string>
namespace cuda_ops_core {
struct Status {

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

  Status();

  Status(Type t, cudaError_t code, std::string msg, std::string file,
              int line, std::string func);

  [[nodiscard]]
  bool ok() const;

  [[nodiscard]]
  std::string str() const;

  static Status
  make(Type t, cudaError_t code, std::string msg = "",
       const std::source_location &loc = std::source_location::current());

  static Status success() noexcept { return Status(); }

private:
  const char *type_string() const;
};
} // namespace cuda_ops_core
