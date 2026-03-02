#include "felix/status.hpp"
#include "kernels/ampere_sgemm_128x128x16_aligned_wip.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {
FelixStatus ampere_sgemm_128x128_nn_a1b0_kernel_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  constexpr uint32_t BM = 128;
  constexpr uint32_t BN = 128;
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  ampere_sgemm_128x128x16_aligned<<<grid, block, 0, stream>>>(M, N, K, alpha, A,
                                                              B, beta, C);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace felix


REGISTER_KERNEL(ampere_sgemm_128x128_nn_a1b0,
                (felix::KernelEntry{
                    felix::KernelType::SGEMM, "ampere_sgemm_128x128_nn_a1b0",
                    (void *)felix::ampere_sgemm_128x128_nn_a1b0_kernel_launch,
                    false}));