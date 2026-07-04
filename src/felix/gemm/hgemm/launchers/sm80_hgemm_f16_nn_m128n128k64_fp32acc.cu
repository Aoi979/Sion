#include "../detail/sm80_hgemm_f16_nn_accum_launcher.cuh"

namespace felix {

FelixStatus sm80_hgemm_f16_nn_m128n128k64_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = detail::validate_sm80_hgemm_f16_nn_accum<shape_mnk>(
      M, N, K, alpha, beta, A, B, C, "m128n128k64_fp32acc");
  if (!status.ok()) {
    return status;
  }
  return detail::convert_sm80_hgemm_status(
      sm80_hgemm::launch_hgemm_128x128x64_fp32acc(
          const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
          static_cast<int>(N), static_cast<int>(K), stream));
}

} // namespace felix

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n128k64_fp32acc,
    felix::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n128k64_fp32acc",
        felix::sm80_hgemm_f16_nn_m128n128k64_fp32acc_launch, true,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 100,
         .required_dynamic_smem_bytes = sm80_hgemm::kSharedStorageBytes,
         .required_threads_per_block = sm80_hgemm::kThreads},
        felix::detail::sm80_hgemm_f16_nn_accum_metadata<shape_mnk>()));
