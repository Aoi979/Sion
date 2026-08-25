#include "../detail/sm80/launcher.cuh"

namespace cuda_ops_core {

Status sm80_hgemm_f16_nn_m128n128k64_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = detail::sm80::launcher::validate_sm80_hgemm_f16_nn_accum<
      detail::sm80::tile::shape_mnk>(
      M, N, K, alpha, beta, A, B, C, "m128n128k64_fp32acc");
  if (!status.ok()) {
    return status;
  }
  return detail::sm80::launcher::convert_sm80_hgemm_status(
      detail::sm80::runtime::launch_hgemm_128x128x64_fp32acc(
          const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
          static_cast<int>(N), static_cast<int>(K), stream));
}

} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n128k64_fp32acc,
    cuda_ops_core::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n128k64_fp32acc",
        cuda_ops_core::sm80_hgemm_f16_nn_m128n128k64_fp32acc_launch, true,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 100,
         .required_dynamic_smem_bytes =
             cuda_ops_core::detail::sm80::runtime::kSharedStorageBytes,
         .required_threads_per_block = cuda_ops_core::detail::sm80::runtime::kThreads},
        cuda_ops_core::detail::sm80::launcher::
            sm80_hgemm_f16_nn_accum_metadata<
                cuda_ops_core::detail::sm80::tile::shape_mnk>()));
