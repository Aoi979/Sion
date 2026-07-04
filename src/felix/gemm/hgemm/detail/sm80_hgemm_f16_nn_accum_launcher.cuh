#pragma once

#include "sm80_hgemm_f16_nn_accum_runtime.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>
#include <string>

namespace felix::detail {

template <typename Shape>
inline FelixStatus validate_sm80_hgemm_f16_nn_accum(uint32_t M, uint32_t N,
                                                    uint32_t K, float alpha,
                                                    float beta, half const *A,
                                                    half const *B, half *C,
                                                    const char *shape_name) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % Shape::M != 0 || N % Shape::N != 0 || K % Shape::K != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        std::string("sm80 HGEMM ") + shape_name +
            " requires non-null A/B/C and M/N/K aligned to tile shape");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return {};
}

inline FelixStatus convert_sm80_hgemm_status(cudaError_t err) {
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

template <typename Shape>
inline GemmKernelMetadata sm80_hgemm_f16_nn_accum_metadata() {
  return {.layout = KernelLayout::NN,
          .align_m = Shape::M,
          .align_n = Shape::N,
          .align_k = Shape::K,
          .requires_alpha_one_beta_zero = true};
}

} // namespace felix::detail
