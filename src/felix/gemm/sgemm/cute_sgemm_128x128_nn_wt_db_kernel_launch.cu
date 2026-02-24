#include "kernels/cute_sgemm_128x128_wt_db.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>
namespace felix {
FelixStatus cute_sgemm_128x128_nn_wt_db_kernel_launch(
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
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto bStages = Int<2>{};

  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA =
      make_layout(make_shape(bM, bK, bStages),
                  make_stride(bK, Int<1>{}, bM * bK)); // (m,k) -> smem_idx; k-major
  auto sB =
      make_layout(make_shape(bN, bK, bStages)); // (n,k) -> smem_idx; n-major
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
  gemm_device_wt_db<<<dimGrid, dimBlock, 0, stream>>>(
      prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha,
      beta);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix


REGISTER_KERNEL(cute_sgemm_128x128_nn_wt_db,
                (felix::KernelEntry{
                    felix::KernelType::SGEMM, "cute_sgemm_128x128_nn_wt_db",
                    (void *)felix::cute_sgemm_128x128_nn_wt_db_kernel_launch, false}));