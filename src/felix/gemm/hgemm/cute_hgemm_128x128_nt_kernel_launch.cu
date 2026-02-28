#include "kernels/cute_ampere_hgemm_16816.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>
namespace felix {

FelixStatus cute_hgemm_128x128_nt_kernel_launch(uint32_t m, uint32_t n,
                                                uint32_t k, float alpha,
                                                cute::half_t const *A,
                                                cute::half_t const *B,
                                                float beta, cute::half_t *C,
                                                cudaStream_t stream = 0) {
  using namespace cute;
  // mxk nxk mxn
  int ldA = k;
  int ldB = k;
  int ldC = n;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(ldC, Int<1>{}); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};                      // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
      Layout<Shape<_1, _8>>{});                 // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
      Layout<Shape<_1, _8>>{});                 // Val layout  1x8 n-major

  TiledMMA mmaC =
      make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                     Layout<Shape<_2, _2>>{}, // 2x2x1 MMA Atoms
                     Tile<_32, _32, _16>{});  // 32x32x16 Tiled MMA for LDSM

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

  int smem_size = int(sizeof(
      SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  auto kernel_fptr =
      cute_ampere_hgemm_16816<decltype(prob_shape), decltype(cta_tiler), cute::half_t,
                  decltype(dA), decltype(sA), decltype(copyA),
                  decltype(s2r_atom_A), cute::half_t, decltype(dB),
                  decltype(sB), decltype(copyB), decltype(s2r_atom_B),
                  cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
                  decltype(alpha), decltype(beta)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  cudaFuncSetAttribute(kernel_fptr,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
      prob_shape, cta_tiler, A, dA, sA, copyA, s2r_atom_A, B, dB, sB, copyB,
      s2r_atom_B, C, dC, sC, mmaC, alpha, beta);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace felix

REGISTER_KERNEL(cute_hgemm_128x128_nt,
                (felix::KernelEntry{
                    felix::KernelType::HGEMM, "cute_hgemm_128x128_nt",
                    (void *)felix::cute_hgemm_128x128_nt_kernel_launch,
                    false}));
