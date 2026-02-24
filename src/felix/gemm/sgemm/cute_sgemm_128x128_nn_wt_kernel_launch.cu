#include "kernels/cute_sgemm_128x128_wt.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>
namespace felix {
FelixStatus cute_sgemm_128x128_nn_wt_kernel_launch(uint32_t m, uint32_t n, uint32_t k,
                                     float alpha, float const *A,
                                     float const *B, float beta, float *C,
                                     cudaStream_t stream) {
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
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK),
                        LayoutRight{});      // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN, bK)); // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN),
                        LayoutRight{}); // (m,n) -> smem_idx; n-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}),
                        LayoutRight{});                    // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<128>{}, Int<2>{})); // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<2>{}, Int<128>{}), LayoutRight{});
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  cudaError_t err;
  err = cudaFuncSetAttribute(
      gemm_device_warp_tiling<decltype(prob_shape), decltype(cta_tiler), decltype(dA),
                  decltype(sA), decltype(tA), decltype(dB), decltype(sB),
                  decltype(tB), decltype(dC), decltype(sC), decltype(tC)>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(float) * cosize_v<decltype(sC)>);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, err);
  }
  gemm_device_warp_tiling<<<dimGrid, dimBlock,
                            sizeof(float) * cosize_v<decltype(sC)>, stream>>>(
      prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha,
      beta);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix


REGISTER_KERNEL(cute_sgemm_128x128_nn_wt,
                (felix::KernelEntry{
                    felix::KernelType::SGEMM, "cute_sgemm_128x128_nn_wt",
                    (void *)felix::cute_sgemm_128x128_nn_wt_kernel_launch, false}));