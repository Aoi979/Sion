#include "../kernels/sm80_sgemm_f32_nn_m64n64k8_cute.cuh"
#include "cute/swizzle.hpp"
#include "cute/swizzle_layout.hpp"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>
namespace cuda_ops_core {
Status sm80_sgemm_f32_nn_m64n64k8_cute_launch(
    uint32_t m, uint32_t n, uint32_t k, float alpha, float const *A,
    float const *B, float beta, float *C, cudaStream_t stream) {
  auto ldA = k;
  auto ldB = n;
  auto ldC = n;
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define NN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
  auto dC = make_stride(ldC, Int<1>{}); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<8>{};
  auto bStages = Int<2>{};

  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  // auto sA =composition(Swizzle<4,2,4>{},
  //     make_layout(make_shape(bM, bK, bStages))); // (m,k) -> smem_idx;
  //     m-major
  auto sA =
      make_layout(make_shape(bM, bK, bStages)); // (m,k) -> smem_idx; m-major
  auto sB =
      make_layout(make_shape(bN, bK, bStages)); // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(Int<32>{}, Int<32>{}),
                        LayoutRight{}); // (32,32) -> smem_idx; n-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<8>{}, Int<8>{}),
                        LayoutRight{});                   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<64>{}, Int<1>{})); // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<1>{}, Int<32>{}), LayoutRight{});
  dim3 dimBlock(Int<2>{} * size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  cudaError_t err;
  sm80_sgemm_f32_nn_m64n64k8_cute_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha,
      beta);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_sgemm_f32_nn_m64n64k8_cute,
    cuda_ops_core::make_sgemm_kernel("sm80_sgemm_f32_nn_m64n64k8_cute",
                             cuda_ops_core::sm80_sgemm_f32_nn_m64n64k8_cute_launch,
                             false, {.min_cc = 80, .max_cc = 89, .priority = 0},
                             {.layout = cuda_ops_core::KernelLayout::NN}));
