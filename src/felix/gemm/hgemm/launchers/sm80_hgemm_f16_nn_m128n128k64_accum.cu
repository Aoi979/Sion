#include "../kernels/sm80_hgemm_f16_nn_m128n128k64_accum.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {
namespace {

FelixStatus validate_sm80_hgemm(uint32_t M, uint32_t N, uint32_t K, float alpha,
                                float beta, half const *A, half const *B,
                                half *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % shape_mnk::M != 0 || N % shape_mnk::N != 0 ||
      K % shape_mnk::K != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM requires non-null A/B/C and M/N/K aligned to 128/128/64");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return {};
}

FelixStatus convert_sm80_hgemm_status(cudaError_t err) {
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace

FelixStatus sm80_hgemm_f16_nn_m128n128k64_fp16acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_sm80_hgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  return convert_sm80_hgemm_status(sm80_hgemm::launch_hgemm_128x128x64_fp16acc(
      const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream));
}

FelixStatus sm80_hgemm_f16_nn_m128n128k64_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_sm80_hgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  return convert_sm80_hgemm_status(sm80_hgemm::launch_hgemm_128x128x64_fp32acc(
      const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream));
}

} // namespace felix

REGISTER_KERNEL(sm80_hgemm_f16_nn_m128n128k64_fp16acc,
                felix::make_hgemm_kernel(
                    "sm80_hgemm_f16_nn_m128n128k64_fp16acc",
                    felix::sm80_hgemm_f16_nn_m128n128k64_fp16acc_launch, false,
                    {.min_cc = 80, .max_cc = 89, .priority = 80},
                    {.layout = felix::KernelLayout::NN,
                     .align_m = shape_mnk::M,
                     .align_n = shape_mnk::N,
                     .align_k = shape_mnk::K,
                     .requires_alpha_one_beta_zero = true}));

REGISTER_KERNEL(sm80_hgemm_f16_nn_m128n128k64_fp32acc,
                felix::make_hgemm_kernel(
                    "sm80_hgemm_f16_nn_m128n128k64_fp32acc",
                    felix::sm80_hgemm_f16_nn_m128n128k64_fp32acc_launch, true,
                    {.min_cc = 80, .max_cc = 89, .priority = 100},
                    {.layout = felix::KernelLayout::NN,
                     .align_m = shape_mnk::M,
                     .align_n = shape_mnk::N,
                     .align_k = shape_mnk::K,
                     .requires_alpha_one_beta_zero = true}));
