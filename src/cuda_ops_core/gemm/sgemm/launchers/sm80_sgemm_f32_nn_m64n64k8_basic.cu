#include "../kernels/sm80_sgemm_f32_nn_m64n64k8_basic.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>
namespace cuda_ops_core {
Status sm80_sgemm_f32_nn_m64n64k8_basic_launch(
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
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m64n64k8_basic,
    cuda_ops_core::make_sgemm_kernel("sm80_sgemm_f32_nn_m64n64k8_basic",
                             cuda_ops_core::sm80_sgemm_f32_nn_m64n64k8_basic_launch,
                             true, {.min_cc = 80, .max_cc = 89, .priority = 10},
                             {.layout = cuda_ops_core::KernelLayout::NN}));
