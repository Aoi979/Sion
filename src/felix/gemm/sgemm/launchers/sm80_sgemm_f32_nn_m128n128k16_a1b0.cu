#include "../kernels/sm80_sgemm_f32_nn_m128n128k16_a1b0.cuh"
#include "felix/status.hpp"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {
FelixStatus sm80_sgemm_f32_nn_m128n128k16_a1b0_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  constexpr uint32_t BM = 128;
  constexpr uint32_t BN = 128;
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sm80_sgemm_f32_nn_m128n128k16_a1b0_kernel<<<grid, block, 0, stream>>>(
      M, N, K, alpha, A, B, beta, C);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace felix

REGISTER_KERNEL(sm80_sgemm_f32_nn_m128n128k16_a1b0,
                felix::make_sgemm_kernel(
                    "sm80_sgemm_f32_nn_m128n128k16_a1b0",
                    felix::sm80_sgemm_f32_nn_m128n128k16_a1b0_launch, false,
                    {.min_cc = 80, .max_cc = 89, .priority = 40},
                    {.layout = felix::KernelLayout::NN,
                     .align_m = 128,
                     .align_n = 128,
                     .align_k = 16,
                     .requires_alpha_one_beta_zero = true}));
