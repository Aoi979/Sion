#include "../kernels/sm80_sgemm_f32_nn_m128n128k8_stage5.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {
namespace {

FelixStatus validate_sm80_stage5_sgemm(uint32_t M, uint32_t N, uint32_t K,
                                       float alpha, float beta, float const *A,
                                       float const *B, float *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % cutlass_like::kCtaM != 0 || N % cutlass_like::kCtaN != 0 ||
      K % cutlass_like::kCtaK != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 stage5 SGEMM requires non-null A/B/C and M/N/K aligned to "
        "128/128/8");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 stage5 SGEMM implements C = A * B only; require alpha=1 and "
        "beta=0");
  }
  return {};
}

FelixStatus check_sm80_stage5_sgemm(cudaError_t err) {
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status = validate_sm80_stage5_sgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  cutlass_like::launch_sgemm_128x128x8stage5(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return check_sm80_stage5_sgemm(cudaSuccess);
}

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status = validate_sm80_stage5_sgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  auto kernel_fptr =
      cutlass_like::sgemm_128x128x8stage5_kernel<false, false, false>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      cutlass_like::kOneCtaPerSmSmemBytes);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, err);
  }
  err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, err);
  }
  cutlass_like::launch_sgemm_128x128x8stage5_one_cta_per_sm(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return check_sm80_stage5_sgemm(cudaSuccess);
}

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_warp_order_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status = validate_sm80_stage5_sgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_warp_order(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return check_sm80_stage5_sgemm(cudaSuccess);
}

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_schedule_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status = validate_sm80_stage5_sgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_schedule(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return check_sm80_stage5_sgemm(cudaSuccess);
}

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status = validate_sm80_stage5_sgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_copy_schedule(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return check_sm80_stage5_sgemm(cudaSuccess);
}

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_sm80_mma_order_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status = validate_sm80_stage5_sgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_sm80_mma_order(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return check_sm80_stage5_sgemm(cudaSuccess);
}

} // namespace felix

REGISTER_KERNEL(sm80_sgemm_f32_nn_m128n128k8_stage5,
                felix::make_sgemm_kernel(
                    "sm80_sgemm_f32_nn_m128n128k8_stage5",
                    felix::sm80_sgemm_f32_nn_m128n128k8_stage5_launch, true,
                    {.min_cc = 80, .max_cc = 89, .priority = 100},
                    {.layout = felix::KernelLayout::NN,
                     .align_m = cutlass_like::kCtaM,
                     .align_n = cutlass_like::kCtaN,
                     .align_k = cutlass_like::kCtaK,
                     .requires_alpha_one_beta_zero = true}));

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm",
        felix::sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm_launch, false,
        {.min_cc = 80, .max_cc = 89, .priority = 60},
        {.layout = felix::KernelLayout::NN,
         .align_m = cutlass_like::kCtaM,
         .align_n = cutlass_like::kCtaN,
         .align_k = cutlass_like::kCtaK,
         .requires_alpha_one_beta_zero = true}));

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_warp_order,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_warp_order",
        felix::sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_warp_order_launch,
        false, {.min_cc = 80, .max_cc = 89, .priority = 60},
        {.layout = felix::KernelLayout::NN,
         .align_m = cutlass_like::kCtaM,
         .align_n = cutlass_like::kCtaN,
         .align_k = cutlass_like::kCtaK,
         .requires_alpha_one_beta_zero = true}));

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_schedule,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_schedule",
        felix::sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_schedule_launch,
        false, {.min_cc = 80, .max_cc = 89, .priority = 60},
        {.layout = felix::KernelLayout::NN,
         .align_m = cutlass_like::kCtaM,
         .align_n = cutlass_like::kCtaN,
         .align_k = cutlass_like::kCtaK,
         .requires_alpha_one_beta_zero = true}));

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule",
        felix::sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule_launch,
        false, {.min_cc = 80, .max_cc = 89, .priority = 60},
        {.layout = felix::KernelLayout::NN,
         .align_m = cutlass_like::kCtaM,
         .align_n = cutlass_like::kCtaN,
         .align_k = cutlass_like::kCtaK,
         .requires_alpha_one_beta_zero = true}));

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_sm80_mma_order,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_sm80_mma_order",
        felix::
            sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_sm80_mma_order_launch,
        false, {.min_cc = 80, .max_cc = 89, .priority = 60},
        {.layout = felix::KernelLayout::NN,
         .align_m = cutlass_like::kCtaM,
         .align_n = cutlass_like::kCtaN,
         .align_k = cutlass_like::kCtaK,
         .requires_alpha_one_beta_zero = true}));
