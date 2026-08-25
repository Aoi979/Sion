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

  // The kernel maps blockIdx.x to the M tile and blockIdx.y to the N tile.
  dim3 grid(static_cast<unsigned>(M / 256), static_cast<unsigned>(N / 256), 1);
  dim3 block(512, 1, 1);

  hgemm_tn_256x256x64_4stage_fp16<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return mcGetLastError();
}

} // namespace cuda_ops_core::maca
