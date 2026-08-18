#include <cuda_ops_core/status.hpp>
#include <sstream>

namespace cuda_ops_core {

Status::Status()
    : type(Type::SUCCESS), cuda_code(cudaSuccess), line(0) {}

Status::Status(Type t, cudaError_t code, std::string msg,
                         std::string f, int l, std::string fn)
    : type(t), cuda_code(code), message(std::move(msg)), file(std::move(f)),
      line(l), func(std::move(fn)) {}

bool Status::ok() const { return type == Type::SUCCESS; }

std::string Status::str() const {
  if (ok())
    return "SUCCESS";

  std::ostringstream oss;
  oss << type_string() << " | cudaError: " << static_cast<int>(cuda_code)
      << " (" << cudaGetErrorString(cuda_code) << ")"
      << " | " << message << " | " << file << ":" << line << " (" << func
      << ")";

  return oss.str();
}

Status Status::make(Type t, cudaError_t code, std::string msg,
                              const std::source_location &loc) {
  return Status(t, code, std::move(msg), loc.file_name(),
                     static_cast<int>(loc.line()), loc.function_name());
}

const char *Status::type_string() const {
  switch (type) {
  case Type::SUCCESS:
    return "SUCCESS";
  case Type::KERNEL_LAUNCH_FAILED:
    return "KERNEL_LAUNCH_FAILED";
  case Type::KERNEL_RUNTIME_ERROR:
    return "KERNEL_RUNTIME_ERROR";
  case Type::API_ERROR:
    return "API_ERROR";
  default:
    return "UNKNOWN";
  }
}
} // namespace cuda_ops_core
