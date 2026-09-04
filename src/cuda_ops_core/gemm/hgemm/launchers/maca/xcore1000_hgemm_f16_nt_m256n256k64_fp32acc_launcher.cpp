#include "../../kernels/maca/xcore1000_hgemm_f16_nt_m256n256k64_fp32acc.hpp"
#include <mc_runtime.h>

namespace cuda_ops_core::maca {

mcError_t xcore1000_hgemm_f16_nt_m256n256k64_fp32acc_launch(const void *A,
                                                            const void *B,
                                                            void *C, int M,
                                                            int N, int K,
                                                            mcStream_t stream) {

  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || N <= 0 ||
      K < 64 || (M % 256) != 0 || (N % 256) != 0 || (K % 64) != 0) {
    return mcErrorInvalidValue;
  }

  // Keep the validated production configuration in one place: the kernel
  // uses the project's CTA swizzle=8 and staggerU=64 workaround.
  constexpr int kCtaSwizzle = 8;
  constexpr int kStaggerU = 64;
  unsigned const tile_m_count = static_cast<unsigned>(M / 256);
  unsigned const tile_n_count = static_cast<unsigned>(N / 256);
  unsigned const grid_x = tile_m_count * kCtaSwizzle;
  unsigned const grid_y =
      (tile_n_count + kCtaSwizzle - 1) / kCtaSwizzle;
  dim3 grid(grid_x, grid_y, 1);
  dim3 block(512, 1, 1);

  hgemm_tn_256x256x64_4stage_fp16<kCtaSwizzle>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, kStaggerU);
  return mcGetLastError();
}

} // namespace cuda_ops_core::maca
