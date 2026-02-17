#include "kernels/ampere_sgemm_64x64_align1.cuh"
#include "kernels/cute_sgemm.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>
namespace felix {
FelixStatus ampere_sgemm_64x64_nn_kernel_launch(uint32_t M, uint32_t N,
                                                uint32_t K, float alpha,
                                                float const *A, float const *B,
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
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix

REGISTER_KERNEL(ampere_sgemm_64x64_nn,
                (felix::KernelEntry{
                    felix::KernelType::SGEMM, "ampere_sgemm_64x64_nn",
                    (void *)felix::ampere_sgemm_64x64_nn_kernel_launch,
                    false}));

