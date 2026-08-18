#include "../kernels/cuda_topk_f32_radix_select.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>

namespace cuda_ops_core {

Status cuda_topk_f32_radix_select_launch(const float *data, float *out,
                                               uint32_t num_slices,
                                               uint32_t slice_size, uint32_t k,
                                               bool largest,
                                               cudaStream_t stream) {
  if (num_slices == 0) {
    return {};
  }
  if (data == nullptr || out == nullptr) {
    return Status::make(Status::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "data/out must be non-null");
  }
  if (slice_size == 0 || k == 0 || k > slice_size) {
    return Status::make(Status::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "k must be in [1, slice_size] and slice_size > 0");
  }

  constexpr uint32_t kThreads = 256;
  dim3 block(kThreads);
  dim3 grid(num_slices);

  cuda_topk_f32_radix_select_kernel<<<grid, block, 0, stream>>>(
      data, out, num_slices, slice_size, k, largest);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace cuda_ops_core

REGISTER_KERNEL(
    cuda_topk_f32_radix_select,
    cuda_ops_core::make_topk_kernel("cuda_topk_f32_radix_select",
                            cuda_ops_core::cuda_topk_f32_radix_select_launch,
                            false));
