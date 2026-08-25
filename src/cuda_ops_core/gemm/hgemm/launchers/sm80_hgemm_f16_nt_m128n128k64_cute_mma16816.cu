#include "../kernels/sm80_hgemm_f16_m128n128k64_cute_mma16816.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>
namespace cuda_ops_core {

Status sm80_hgemm_f16_nt_m128n128k64_cute_mma16816_launch(
    uint32_t m, uint32_t n, uint32_t k, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream = 0) {
  using namespace cute;
  auto cute_A = reinterpret_cast<cute::half_t const *>(A);
  auto cute_B = reinterpret_cast<cute::half_t const *>(B);
  auto cute_C = reinterpret_cast<cute::half_t *>(C);

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

  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

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

  int smem_size =
      int(sizeof(CuteHgemmSharedStorage<cute::half_t, cute::half_t,
                                        decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  constexpr int kBlockSwizzle = 8;
  int tile_m_count = size(ceil_div(M, bM));
  int tile_n_count = size(ceil_div(N, bN));
  dim3 dimGrid(tile_m_count * kBlockSwizzle,
               (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);

  auto kernel_fptr = sm80_hgemm_f16_m128n128k64_cute_mma16816_kernel<
      decltype(prob_shape), decltype(cta_tiler), cute::half_t, decltype(dA),
      decltype(sA), decltype(copyA), decltype(s2r_atom_A), cute::half_t,
      decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
      cute::half_t, decltype(dC), decltype(sC), decltype(mmaC), decltype(alpha),
      decltype(beta)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  cudaFuncSetAttribute(kernel_fptr,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
      prob_shape, cta_tiler, cute_A, dA, sA, copyA, s2r_atom_A, cute_B, dB, sB,
      copyB, s2r_atom_B, cute_C, dC, sC, mmaC, alpha, beta, kBlockSwizzle);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}
} // namespace cuda_ops_core

REGISTER_KERNEL(sm80_hgemm_f16_nt_m128n128k64_cute_mma16816,
                cuda_ops_core::make_hgemm_kernel(
                    "sm80_hgemm_f16_nt_m128n128k64_cute_mma16816",
                    cuda_ops_core::sm80_hgemm_f16_nt_m128n128k64_cute_mma16816_launch,
                    true, {.min_cc = 80, .max_cc = 89, .priority = 50},
                    {.layout = cuda_ops_core::KernelLayout::NT,
                     .align_m = 128,
                     .align_n = 128,
                     .align_k = 64,
                     .requires_alpha_one_beta_zero = true}));
