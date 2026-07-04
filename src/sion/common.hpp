#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <cstdint>

namespace sion::detail {

inline uint32_t checked_u32(int64_t value, const char *name) {
  TORCH_CHECK(value >= 0 && value <= static_cast<int64_t>(UINT32_MAX), name,
              " must fit in uint32_t, got ", value);
  return static_cast<uint32_t>(value);
}

inline void check_same_cuda_device(const torch::Tensor &a,
                                   const torch::Tensor &b, const char *a_name,
                                   const char *b_name) {
  TORCH_CHECK(a.is_cuda(), a_name, " must be CUDA tensor");
  TORCH_CHECK(b.is_cuda(), b_name, " must be CUDA tensor");
  TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name,
              " must be on the same CUDA device");
}

} // namespace sion::detail
