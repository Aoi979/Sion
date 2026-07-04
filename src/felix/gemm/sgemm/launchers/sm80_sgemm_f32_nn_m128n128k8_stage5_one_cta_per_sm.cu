#include "../detail/sm80_sgemm_f32_nn_m128n128k8_stage5_launcher.cuh"

namespace felix {

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status =
      detail::validate_sm80_sgemm_f32_nn_m128n128k8_stage5(M, N, K, alpha, beta,
                                                           A, B, C);
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
  return detail::check_sm80_sgemm_f32_nn_m128n128k8_stage5(cudaSuccess);
}

} // namespace felix

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm",
        felix::sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm_launch, false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 60,
         .required_dynamic_smem_bytes = cutlass_like::kOneCtaPerSmSmemBytes,
         .required_threads_per_block = cutlass_like::kThreads},
        felix::detail::sm80_sgemm_f32_nn_m128n128k8_stage5_metadata()));
