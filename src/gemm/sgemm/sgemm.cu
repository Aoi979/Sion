#include "ampere_sgemm_64x64_modular.cuh"
#include <felix/felix.hpp>
namespace felix {
FelixStatus ampere_sgemm_64x64_nn_kernel_launch(uint32_t M, uint32_t N, uint32_t K,
                                         float alpha,
                                         float const *A,
                                         float const *B,
                                         float beta, float *C,
                                         cudaStream_t stream) {
  constexpr uint32_t BM = 64;
  constexpr uint32_t BN = 64;
  dim3 block(64);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  ampere_sgemm_64x64_nn<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta,
                                                    C);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(
        FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix