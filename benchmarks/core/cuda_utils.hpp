#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cuda_ops::bench {

inline void cuda_check(cudaError_t err, const std::string &where) {
  if (err != cudaSuccess) {
    throw std::runtime_error(where + ": " + cudaGetErrorString(err));
  }
}

struct DeviceInfo {
  int ordinal = 0;
  std::string name;
  int cc = 0;
  int sm_count = 0;
  std::uint32_t max_dynamic_smem = 0;
  std::uint32_t max_threads_per_block = 0;
};

inline DeviceInfo current_device_info() {
  DeviceInfo info;
  cuda_check(cudaGetDevice(&info.ordinal), "cudaGetDevice");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, info.ordinal),
             "cudaGetDeviceProperties");

  info.name = prop.name;
  info.cc = prop.major * 10 + prop.minor;
  info.sm_count = prop.multiProcessorCount;
  const auto default_smem =
      static_cast<std::uint32_t>(prop.sharedMemPerBlock);
  const auto optin_smem =
      static_cast<std::uint32_t>(prop.sharedMemPerBlockOptin);
  info.max_dynamic_smem =
      default_smem > optin_smem ? default_smem : optin_smem;
  info.max_threads_per_block =
      static_cast<std::uint32_t>(prop.maxThreadsPerBlock);
  return info;
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { reset(count); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  void reset(std::size_t count) {
    release();
    count_ = count;
    if (count_ != 0) {
      cuda_check(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
    }
  }

  void memset_zero(cudaStream_t stream) {
    if (ptr_ != nullptr) {
      cuda_check(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), stream),
                 "cudaMemsetAsync");
    }
  }

  T *get() const { return ptr_; }
  std::size_t count() const { return count_; }

private:
  void release() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  T *ptr_ = nullptr;
  std::size_t count_ = 0;
};

} // namespace cuda_ops::bench
