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

FelixStatus cute_gemm_nn(uint32_t m, uint32_t n, uint32_t k, float alpha,
                         float const *A, float const *B, float beta, float *C,
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
                        LayoutRight{});                   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{})); // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}),
                        LayoutRight{}); // (m,n) -> thr_idx

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                sA, tA, B, dB, sB, tB, C, dC,
                                                sC, tC, alpha, beta);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace felix

REGISTER_KERNEL(ampere_sgemm_64x64_nn,
                (felix::KernelEntry{felix::KernelType::SGEMM,
                                    "ampere_sgemm_64x64_nn",
                                    (void *)felix::cute_gemm_nn, false}));
// REGISTER_KERNEL(cute_gemm_nn, (felix::KernelEntry{
//                              felix::KernelType::SGEMM, "cute_gemm_nn",
//                              (void *)felix::cute_gemm_nn,
//                              false}));
