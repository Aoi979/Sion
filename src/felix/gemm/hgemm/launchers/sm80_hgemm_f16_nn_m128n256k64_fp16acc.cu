#include "../detail/sm80_hgemm_f16_nn_accum_launcher.cuh"

namespace felix {

FelixStatus sm80_hgemm_f16_nn_m128n256k64_fp16acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = detail::validate_sm80_hgemm_f16_nn_accum<shape_mnk_n256>(
      M, N, K, alpha, beta, A, B, C, "m128n256k64_fp16acc");
  if (!status.ok()) {
    return status;
  }
  return detail::convert_sm80_hgemm_status(
      sm80_hgemm::launch_hgemm_128x256x64_fp16acc(
          const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
          static_cast<int>(N), static_cast<int>(K), stream));
}

} // namespace felix

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n256k64_fp16acc,
    felix::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n256k64_fp16acc",
        felix::sm80_hgemm_f16_nn_m128n256k64_fp16acc_launch, false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 70,
         .required_dynamic_smem_bytes = sm80_hgemm::kSharedStorageBytesN256,
         .required_threads_per_block = sm80_hgemm::kThreadsN256},
        felix::detail::sm80_hgemm_f16_nn_accum_metadata<shape_mnk_n256>()));
