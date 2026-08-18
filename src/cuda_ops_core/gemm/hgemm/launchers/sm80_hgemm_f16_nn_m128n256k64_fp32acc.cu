#include "../detail/sm80_hgemm_f16_nn_m128n256k64_fp32acc_runtime.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>
#include <string>

namespace cuda_ops_core {

Status sm80_hgemm_f16_nn_m128n256k64_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 ||
      M % detail::sm80_hgemm_128x256_fp32acc::shape_mnk_n256::M != 0 ||
      N % detail::sm80_hgemm_128x256_fp32acc::shape_mnk_n256::N != 0 ||
      K % detail::sm80_hgemm_128x256_fp32acc::shape_mnk_n256::K != 0) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM m128n256k64_fp32acc requires non-null A/B/C and M/N/K "
        "aligned to tile shape");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }

  int block_swizzle = detail::sm80_hgemm_128x256_fp32acc::select_block_swizzle(
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
  cudaError_t err =
      detail::sm80_hgemm_128x256_fp32acc::launch_hgemm_128x256x64_fp32acc(
          const_cast<half *>(A), const_cast<half *>(B), C,
          static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
          block_swizzle, stream);
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n256k64_fp32acc,
    cuda_ops_core::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n256k64_fp32acc",
        cuda_ops_core::sm80_hgemm_f16_nn_m128n256k64_fp32acc_launch, false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 95,
         .required_dynamic_smem_bytes =
             cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc::kSharedStorageBytes,
         .required_threads_per_block =
             cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc::kThreads},
        {.layout = cuda_ops_core::KernelLayout::NN,
         .align_m =
             cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc::shape_mnk_n256::M,
         .align_n =
             cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc::shape_mnk_n256::N,
         .align_k =
             cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc::shape_mnk_n256::K,
         .requires_alpha_one_beta_zero = true}));
