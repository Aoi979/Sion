#include "../detail/sm80_sgemm_f32_nn_m128n128k8_stage5_launcher.cuh"

namespace felix {

FelixStatus sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto status =
      detail::validate_sm80_sgemm_f32_nn_m128n128k8_stage5(M, N, K, alpha, beta,
                                                           A, B, C);
  if (!status.ok()) {
    return status;
  }
  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_copy_schedule(
      const_cast<float *>(A), const_cast<float *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream);
  return detail::check_sm80_sgemm_f32_nn_m128n128k8_stage5(cudaSuccess);
}

} // namespace felix

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule,
    felix::make_sgemm_kernel(
        "sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule",
        felix::sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule_launch,
        false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 60,
         .required_dynamic_smem_bytes = cutlass_like::kSharedStorageBytes,
         .required_threads_per_block = cutlass_like::kThreads},
        felix::detail::sm80_sgemm_f32_nn_m128n128k8_stage5_metadata()));
