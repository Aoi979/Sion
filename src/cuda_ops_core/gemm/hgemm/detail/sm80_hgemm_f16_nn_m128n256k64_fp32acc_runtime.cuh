#pragma once

#include "../kernels/sm80_hgemm_f16_nn_m128n256k64_fp32acc.cuh"

namespace cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc {

constexpr int kStages = 3;
constexpr int kDefaultBlockSwizzle = 8;
constexpr int kAutoBlockSwizzle = 0;
constexpr int kThreads = 256;
constexpr int kSharedStorageBytes =
    sizeof(HgemmSharedStorage<shape_mnk_n256, kStages>);

inline int select_block_swizzle(int M, int N, int K) {
  (void)M;
  (void)K;
  int const tile_n_count =
      (N + shape_mnk_n256::N - 1) / shape_mnk_n256::N;
  return tile_n_count <= 16 ? 1 : kDefaultBlockSwizzle;
}

template <int BlockSwizzle>
inline cudaError_t configure_hgemm_128x256x64_fp32acc() {
  auto kernel_fptr = n256::sm80_hgemm_f16_nn_m128n256k64_fp32acc_kernel<
      shape_mnk_n256, kStages, BlockSwizzle>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageBytes);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaFuncSetAttribute(kernel_fptr,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              100);
}

inline cudaError_t configure_hgemm_128x256x64_fp32acc(int block_swizzle) {
  switch (block_swizzle) {
  case 1:
    return configure_hgemm_128x256x64_fp32acc<1>();
  case 2:
    return configure_hgemm_128x256x64_fp32acc<2>();
  case 4:
    return configure_hgemm_128x256x64_fp32acc<4>();
  case 8:
    return configure_hgemm_128x256x64_fp32acc<8>();
  case 16:
    return configure_hgemm_128x256x64_fp32acc<16>();
  case 32:
    return configure_hgemm_128x256x64_fp32acc<32>();
  case 64:
    return configure_hgemm_128x256x64_fp32acc<64>();
  default:
    return cudaErrorInvalidValue;
  }
}

template <int BlockSwizzle>
inline void launch_hgemm_128x256x64_fp32acc_unchecked(
    half *A, half *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  int tile_m_count = M / shape_mnk_n256::M;
  int tile_n_count = N / shape_mnk_n256::N;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * BlockSwizzle,
            (tile_n_count + BlockSwizzle - 1) / BlockSwizzle);
  n256::sm80_hgemm_f16_nn_m128n256k64_fp32acc_kernel<
      shape_mnk_n256, kStages, BlockSwizzle>
      <<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
}

inline void launch_hgemm_128x256x64_fp32acc_unchecked(
    half *A, half *B, half *C, int M, int N, int K, int block_swizzle,
    cudaStream_t stream = 0) {
  switch (block_swizzle) {
  case 1:
    launch_hgemm_128x256x64_fp32acc_unchecked<1>(A, B, C, M, N, K, stream);
    return;
  case 2:
    launch_hgemm_128x256x64_fp32acc_unchecked<2>(A, B, C, M, N, K, stream);
    return;
  case 4:
    launch_hgemm_128x256x64_fp32acc_unchecked<4>(A, B, C, M, N, K, stream);
    return;
  case 8:
    launch_hgemm_128x256x64_fp32acc_unchecked<8>(A, B, C, M, N, K, stream);
    return;
  case 16:
    launch_hgemm_128x256x64_fp32acc_unchecked<16>(A, B, C, M, N, K, stream);
    return;
  case 32:
    launch_hgemm_128x256x64_fp32acc_unchecked<32>(A, B, C, M, N, K, stream);
    return;
  case 64:
    launch_hgemm_128x256x64_fp32acc_unchecked<64>(A, B, C, M, N, K, stream);
    return;
  default:
    return;
  }
}

inline cudaError_t launch_hgemm_128x256x64_fp32acc(
    half *A, half *B, half *C, int M, int N, int K, int block_swizzle,
    cudaStream_t stream = 0) {
  cudaError_t err = configure_hgemm_128x256x64_fp32acc(block_swizzle);
  if (err != cudaSuccess) {
    return err;
  }
  launch_hgemm_128x256x64_fp32acc_unchecked(A, B, C, M, N, K, block_swizzle,
                                            stream);
  return cudaGetLastError();
}

inline cudaError_t launch_hgemm_128x256x64_fp32acc(
    half *A, half *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  return launch_hgemm_128x256x64_fp32acc(
      A, B, C, M, N, K, kAutoBlockSwizzle, stream);
}

} // namespace cuda_ops_core::detail::sm80_hgemm_128x256_fp32acc
