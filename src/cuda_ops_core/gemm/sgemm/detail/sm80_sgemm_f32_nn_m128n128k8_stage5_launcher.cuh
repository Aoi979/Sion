#pragma once

#include "../kernels/sm80_sgemm_f32_nn_m128n128k8_stage5.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>

namespace cuda_ops_core::detail {

inline Status validate_sm80_sgemm_f32_nn_m128n128k8_stage5(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float beta, float const *A,
    float const *B, float *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % cutlass_like::kCtaM != 0 || N % cutlass_like::kCtaN != 0 ||
      K % cutlass_like::kCtaK != 0) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 stage5 SGEMM requires non-null A/B/C and M/N/K aligned to "
        "128/128/8");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 stage5 SGEMM implements C = A * B only; require alpha=1 and "
        "beta=0");
  }
  return {};
}

inline Status check_sm80_sgemm_f32_nn_m128n128k8_stage5(cudaError_t err) {
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

inline GemmKernelMetadata sm80_sgemm_f32_nn_m128n128k8_stage5_metadata() {
  return {.layout = KernelLayout::NN,
          .align_m = cutlass_like::kCtaM,
          .align_n = cutlass_like::kCtaN,
          .align_k = cutlass_like::kCtaK,
          .requires_alpha_one_beta_zero = true};
}

} // namespace cuda_ops_core::detail
