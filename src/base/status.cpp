#include <felix/status.hpp>
#include <sstream>

namespace felix {

FelixStatus::FelixStatus()
    : type(Type::SUCCESS), cuda_code(cudaSuccess), line(0) {}

FelixStatus::FelixStatus(Type t, cudaError_t code, std::string msg,
                         std::string f, int l, std::string fn)
    : type(t), cuda_code(code), message(std::move(msg)), file(std::move(f)),
      line(l), func(std::move(fn)) {}

bool FelixStatus::ok() const { return type == Type::SUCCESS; }

std::string FelixStatus::str() const {
  if (ok())
    return "SUCCESS";

  std::ostringstream oss;
  oss << type_string() << " | cudaError: " << static_cast<int>(cuda_code)
      << " (" << cudaGetErrorString(cuda_code) << ")"
      << " | " << message << " | " << file << ":" << line << " (" << func
      << ")";

  return oss.str();
}

FelixStatus FelixStatus::make(Type t, cudaError_t code, std::string msg,
                              const std::source_location &loc) {
  return FelixStatus(t, code, std::move(msg), loc.file_name(),
                     static_cast<int>(loc.line()), loc.function_name());
}

const char *FelixStatus::type_string() const {
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
} // namespace felix
