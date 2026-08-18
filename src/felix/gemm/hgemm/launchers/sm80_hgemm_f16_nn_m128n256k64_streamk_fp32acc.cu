#include "../kernels/sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace felix {
namespace {

namespace streamk = detail::sm80_hgemm_128x256_streamk::n256_streamk;
using StreamKShape = detail::sm80_hgemm_128x256_streamk::shape_mnk_n256;

FelixStatus validate_streamk_request(uint32_t M, uint32_t N, uint32_t K,
                                     float alpha, float beta, half const *A,
                                     half const *B, half *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % StreamKShape::M != 0 || N % StreamKShape::N != 0 ||
      K % StreamKShape::K != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM m128n256k64_streamk_fp32acc requires non-null A/B/C and "
        "aligned M/N/K");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return FelixStatus{};
}

FelixStatus convert_streamk_error(cudaError_t error) {
  if (error == cudaSuccess) {
    return {};
  }
  return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, error);
}

FelixStatus allocate_partial_workspace(float **partial, size_t elements,
                                       cudaStream_t stream) {
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "sm80 HGEMM Stream-K workspace size overflows");
  }
  cudaError_t error = cudaMallocAsync(
      reinterpret_cast<void **>(partial), elements * sizeof(float), stream);
  if (error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, error,
                             "failed to allocate Stream-K workspace");
  }
  return {};
}

FelixStatus release_partial_workspace(float *partial, cudaStream_t stream,
                                      FelixStatus status) {
  if (partial == nullptr) {
    return status;
  }
  cudaError_t error = cudaFreeAsync(partial, stream);
  if (status.ok() && error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, error,
                             "failed to release Stream-K workspace");
  }
  return status;
}

} // namespace

FelixStatus sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_streamk_request(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, error);
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, error);
  }

  auto plan = streamk::make_streamk_schedule_plan(
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), 4,
      properties.multiProcessorCount);
  if (!plan.valid()) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM Stream-K could not create a valid schedule");
  }
  auto params = streamk::make_streamk_params(plan);
  if (!params.valid()) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM Stream-K produced invalid launch parameters");
  }

  float *partial = nullptr;
  if (plan.partials_elements != 0) {
    status = allocate_partial_workspace(&partial, plan.partials_elements,
                                        stream);
    if (!status.ok()) {
      return status;
    }
  }

  error = streamk::configure_hgemm_128x256_streamk_fp32acc();
  if (error == cudaSuccess) {
    streamk::launch_hgemm_128x256_streamk_fp32acc_unchecked(
        A, B, partial, C, static_cast<int>(N), static_cast<int>(K), params,
        stream);
    error = cudaGetLastError();
  }
  return release_partial_workspace(partial, stream,
                                   convert_streamk_error(error));
}

} // namespace felix

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc,
    felix::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc",
        felix::sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_launch, false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 90,
         .required_dynamic_smem_bytes =
             felix::detail::sm80_hgemm_128x256_streamk::
                 n256_streamk::kStreamKSharedStorageBytes128x256,
         .required_threads_per_block =
             felix::detail::sm80_hgemm_128x256_streamk::n256_streamk::
                 kStreamKThreads},
        {.layout = felix::KernelLayout::NN,
         .align_m = 128,
         .align_n = 256,
         .align_k = 128,
         .requires_alpha_one_beta_zero = true}));
