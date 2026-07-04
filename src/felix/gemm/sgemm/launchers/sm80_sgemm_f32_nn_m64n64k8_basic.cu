#include "../kernels/sm80_sgemm_f32_nn_m64n64k8_basic.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>
namespace felix {
FelixStatus sm80_sgemm_f32_nn_m64n64k8_basic_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  constexpr uint32_t BM = 64;
  constexpr uint32_t BN = 64;
  dim3 block(64);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sm80_sgemm_f32_nn_m64n64k8_basic_kernel<<<grid, block, 0, stream>>>(
      M, N, K, alpha, A, B, beta, C);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m64n64k8_basic,
    felix::make_sgemm_kernel("sm80_sgemm_f32_nn_m64n64k8_basic",
                             felix::sm80_sgemm_f32_nn_m64n64k8_basic_launch,
                             true, {.min_cc = 80, .max_cc = 89, .priority = 10},
                             {.layout = felix::KernelLayout::NN}));
