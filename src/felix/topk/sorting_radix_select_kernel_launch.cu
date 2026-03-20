#include "kernels/sorting_radix_select.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {

FelixStatus sorting_radix_select_kernel_launch(const float *data, float *out,
                                               uint32_t num_slices,
                                               uint32_t slice_size, uint32_t k,
                                               bool largest,
                                               cudaStream_t stream) {
  if (num_slices == 0) {
    return {};
  }
  if (data == nullptr || out == nullptr) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "data/out must be non-null");
  }
  if (slice_size == 0 || k == 0 || k > slice_size) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "k must be in [1, slice_size] and slice_size > 0");
  }

  constexpr uint32_t kThreads = 256;
  dim3 block(kThreads);
  dim3 grid(num_slices);

  sorting_radix_select_kernel<<<grid, block, 0, stream>>>(
      data, out, num_slices, slice_size, k, largest);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix

REGISTER_KERNEL(
    sorting_radix_select,
    (felix::KernelEntry{felix::KernelType::TopK, "sorting_radix_select",
                        (void *)felix::sorting_radix_select_kernel_launch,
                        false}));
