#pragma once

#include "../kernels/sm80_hgemm_f16_nn_m128n128k64_fp16acc.cuh"
#include "../kernels/sm80_hgemm_f16_nn_m128n128k64_fp32acc.cuh"
#include "../kernels/sm80_hgemm_f16_nn_m128n256k64_fp16acc.cuh"

namespace sm80_hgemm {

constexpr int kStages = 3;
constexpr int kBlockSwizzle = 8;
constexpr int kAutoBlockSwizzle = 0;
constexpr int kThreads = 128;
constexpr int kThreadsN256 = 256;
constexpr int kSharedStorageBytes =
    sizeof(HgemmSharedStorage<shape_mnk, kStages>);
constexpr int kSharedStorageBytesN256 =
    sizeof(HgemmSharedStorage<shape_mnk_n256, kStages>);

inline int select_hgemm_128x128x64_fp16acc_block_swizzle(int M, int N, int K) {
  (void)M;
  (void)K;
  int const tile_n_count = (N + shape_mnk::N - 1) / shape_mnk::N;
  if (tile_n_count <= 16) {
    return 1;
  }
  return kBlockSwizzle;
}

template <int BlockSwizzle>
inline cudaError_t launch_hgemm_128x128x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  auto kernel_fptr = hgemm_f16f16f16_kernel<shape_mnk, kStages, BlockSwizzle>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageBytes);
  if (err != cudaSuccess)
    return err;

  err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess)
    return err;

  int tile_m_count = M / shape_mnk::M;
  int tile_n_count = N / shape_mnk::N;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * BlockSwizzle,
            (tile_n_count + BlockSwizzle - 1) / BlockSwizzle);

  kernel_fptr<<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

inline cudaError_t launch_hgemm_128x128x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   int block_swizzle,
                                                   cudaStream_t stream = 0) {
  if (block_swizzle == kAutoBlockSwizzle) {
    block_swizzle = select_hgemm_128x128x64_fp16acc_block_swizzle(M, N, K);
  }
  switch (block_swizzle) {
  case 1:
    return launch_hgemm_128x128x64_fp16acc<1>(A, B, C, M, N, K, stream);
  case 2:
    return launch_hgemm_128x128x64_fp16acc<2>(A, B, C, M, N, K, stream);
  case 4:
    return launch_hgemm_128x128x64_fp16acc<4>(A, B, C, M, N, K, stream);
  case 8:
    return launch_hgemm_128x128x64_fp16acc<8>(A, B, C, M, N, K, stream);
  case 16:
    return launch_hgemm_128x128x64_fp16acc<16>(A, B, C, M, N, K, stream);
  case 32:
    return launch_hgemm_128x128x64_fp16acc<32>(A, B, C, M, N, K, stream);
  case 64:
    return launch_hgemm_128x128x64_fp16acc<64>(A, B, C, M, N, K, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

inline cudaError_t launch_hgemm_128x128x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  return launch_hgemm_128x128x64_fp16acc(A, B, C, M, N, K, kAutoBlockSwizzle,
                                         stream);
}

inline int select_hgemm_128x256x64_fp16acc_block_swizzle(int M, int N, int K) {
  (void)M;
  (void)K;
  int const tile_n_count = (N + shape_mnk_n256::N - 1) / shape_mnk_n256::N;
  if (tile_n_count <= 16) {
    return 1;
  }
  return kBlockSwizzle;
}

template <int BlockSwizzle>
inline cudaError_t launch_hgemm_128x256x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  auto kernel_fptr =
      n256::hgemm_f16f16f16_128x256_kernel<shape_mnk_n256, kStages,
                                           BlockSwizzle>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageBytesN256);
  if (err != cudaSuccess)
    return err;

  err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess)
    return err;

  int tile_m_count = M / shape_mnk_n256::M;
  int tile_n_count = N / shape_mnk_n256::N;
  dim3 block(kThreadsN256);
  dim3 grid(tile_m_count * BlockSwizzle,
            (tile_n_count + BlockSwizzle - 1) / BlockSwizzle);

  kernel_fptr<<<grid, block, kSharedStorageBytesN256, stream>>>(A, B, C, M, N,
                                                                K);
  return cudaGetLastError();
}

inline cudaError_t launch_hgemm_128x256x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   int block_swizzle,
                                                   cudaStream_t stream = 0) {
  if (block_swizzle == kAutoBlockSwizzle) {
    block_swizzle = select_hgemm_128x256x64_fp16acc_block_swizzle(M, N, K);
  }
  switch (block_swizzle) {
  case 1:
    return launch_hgemm_128x256x64_fp16acc<1>(A, B, C, M, N, K, stream);
  case 2:
    return launch_hgemm_128x256x64_fp16acc<2>(A, B, C, M, N, K, stream);
  case 4:
    return launch_hgemm_128x256x64_fp16acc<4>(A, B, C, M, N, K, stream);
  case 8:
    return launch_hgemm_128x256x64_fp16acc<8>(A, B, C, M, N, K, stream);
  case 16:
    return launch_hgemm_128x256x64_fp16acc<16>(A, B, C, M, N, K, stream);
  case 32:
    return launch_hgemm_128x256x64_fp16acc<32>(A, B, C, M, N, K, stream);
  case 64:
    return launch_hgemm_128x256x64_fp16acc<64>(A, B, C, M, N, K, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

inline cudaError_t launch_hgemm_128x256x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  return launch_hgemm_128x256x64_fp16acc(A, B, C, M, N, K, kAutoBlockSwizzle,
                                         stream);
}

inline cudaError_t launch_hgemm_128x128x64_fp32acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  auto kernel_fptr = hgemm_f16f16f32_kernel<shape_mnk, kStages, kBlockSwizzle>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageBytes);
  if (err != cudaSuccess)
    return err;

  err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess)
    return err;

  int tile_m_count = M / shape_mnk::M;
  int tile_n_count = N / shape_mnk::N;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);

  kernel_fptr<<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

} // namespace sm80_hgemm
