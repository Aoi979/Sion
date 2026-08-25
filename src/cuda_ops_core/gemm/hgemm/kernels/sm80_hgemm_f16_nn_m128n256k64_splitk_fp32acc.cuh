#pragma once

#include "../detail/sm80/splitk.cuh"

namespace cuda_ops_core::detail::sm80::splitk {

using namespace ::cuda_ops_core::detail::sm80::common;
using namespace ::cuda_ops_core::detail::sm80::tile;
using namespace support;

template <typename Shape_MNK = shape_mnk_n256, int kStages, int kBlockSwizzle>
__global__ void sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_kernel(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k) {
  constexpr int kCtaM = Shape_MNK::M; // 128
  constexpr int kCtaN = Shape_MNK::N; // 256
  constexpr int kCtaK = Shape_MNK::K; // 64
  static_assert(kCtaM == 128 && kCtaN == 256 && kCtaK == 64,
                "swizzled shared-memory layout assumes a 128x256x64 CTA");

  constexpr int kWarpsM = 2;
  constexpr int kWarpSize = 32;

  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 64;
  constexpr int Tiled_MMA_K = 16;

  extern __shared__ char shared_memory[];
  using MainLoopSharedStorage = HgemmSharedStorage<Shape_MNK, kStages>;
  MainLoopSharedStorage *smem =
      reinterpret_cast<MainLoopSharedStorage *>(shared_memory);

  int StrideA = K;
  int StrideB = N;

  int const tile_m_max = (M + kCtaM - 1) / kCtaM;
  int const tile_n_max = (N + kCtaN - 1) / kCtaN;

  int tile_m = blockIdx.x / kBlockSwizzle;
  int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tile_m >= tile_m_max || tile_n >= tile_n_max) {
    return;
  }

  if (split_k <= 0 || K % kCtaK != 0) {
    return;
  }
  int const split_id = static_cast<int>(blockIdx.z);
  int const k_tile_count = K / kCtaK;
  if (split_id >= split_k || k_tile_count % split_k != 0) {
    return;
  }
  int const k_tiles_per_split = k_tile_count / split_k;
  int const k_begin = split_id * k_tiles_per_split * kCtaK;

  const half *gA_base =
      A + tile_m * kCtaM * StrideA + k_begin;
  const half *gB_base =
      B + k_begin * StrideB + tile_n * kCtaN;

  long long const matrix_elements = static_cast<long long>(M) * N;
  float *gPartial =
      partial + static_cast<long long>(split_id) * matrix_elements +
      static_cast<long long>(tile_m) * kCtaM * N +
      static_cast<long long>(tile_n) * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;

  int const K_TILE_MAX = k_tiles_per_split;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;
  static_assert(K_BLOCK_MAX == 4,
                "mainloop cp.async schedule assumes four MMA K-blocks");

  constexpr int MMA_M = kCtaM / Tiled_MMA_M;
  constexpr int MMA_N = kCtaN / Tiled_MMA_N;
  constexpr int MMA_K = kCtaK / Tiled_MMA_K;
  constexpr int kFragmentSlots = 2;
  static_assert(MMA_K == K_BLOCK_MAX,
                "fragment slots assume one logical slot per MMA K-block");

  constexpr int Fragment = 2;
  constexpr int CoreMatrix_M = 2;
  constexpr int CoreMatrix_N = 2;
  constexpr int CoreMatrix_K = 2;

  constexpr int kElementsPerAccess = 8; // half, 16B

  float tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][Fragment];
  half tCrA[kFragmentSlots][MMA_M][CoreMatrix_K][CoreMatrix_M][Fragment];
  half tCrB[kFragmentSlots][MMA_N][CoreMatrix_N][CoreMatrix_K][Fragment];

  int lane_id = tid % kWarpSize;

  int tA_row = tid / (kCtaK / kElementsPerAccess); // 8
  int tA_col = tid % (kCtaK / kElementsPerAccess);

  int tB_row = tid / (kCtaN / kElementsPerAccess); // 32
  int tB_col = tid % (kCtaN / kElementsPerAccess);

  int k_tiles_to_compute = K_TILE_MAX;
  const half *gA_next = gA_base;
  const half *gB_next = gB_base;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    issue_cp_async_A4(smem->buffer[k_pipe].A, gA_next, tA_row, tA_col,
                      StrideA);
    issue_cp_async_B8(smem->buffer[k_pipe].B, gB_next, tB_row, tB_col,
                      StrideB);

    cp_async::commit_group();
    --k_tiles_to_compute;
    if (k_tiles_to_compute > 0) {
      gA_next += kCtaK;
      gB_next += kCtaK * StrideB;
    }
  }

  constexpr int kBufferBytes = sizeof(Buffer<Shape_MNK>);
  constexpr int kAElements = Shape_MNK::M * Shape_MNK::K;
  int smem_read_offset = 0;
  int smem_write_offset = (K_PIPE_MAX - 1) * kBufferBytes;
  char *smem_bytes = reinterpret_cast<char *>(smem);
  half *smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
  half *smem_read_B = smem_read_A + kAElements;

  int warp_m_id = warp_id % kWarpsM;
  int warp_n_id = warp_id / kWarpsM;

  int ldsmx4_row = lane_id % 16;
  int ldsmx4_col = lane_id / 16;

  int ldsmx4T_col = lane_id % 16;
  int ldsmx4T_row = lane_id / 16;

  if constexpr (K_BLOCK_MAX > 1) {
    cp_async::wait_group<K_PIPE_MAX - 2>();
    __syncthreads();

    ldsm::x4<ldsm::N>(as_u32(tCrA[0][0][0][0][0]), as_u32(tCrA[0][0][0][1][0]),
                      as_u32(tCrA[0][0][1][0][0]), as_u32(tCrA[0][0][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][1][0][0][0]), as_u32(tCrA[0][1][0][1][0]),
                      as_u32(tCrA[0][1][1][0][0]), as_u32(tCrA[0][1][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][2][0][0][0]), as_u32(tCrA[0][2][0][1][0]),
                      as_u32(tCrA[0][2][1][0][0]), as_u32(tCrA[0][2][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][3][0][0][0]), as_u32(tCrA[0][3][0][1][0]),
                      as_u32(tCrA[0][3][1][0][0]), as_u32(tCrA[0][3][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));

    half *b_smem = smem_read_B;
    int b_ldsm_base = tile_n256::offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32,
        ldsmx4T_col + 0 * Tiled_MMA_K);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 0]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][1][0][0][0]), as_u32(tCrB[0][1][0][1][0]),
                      as_u32(tCrB[0][1][1][0][0]), as_u32(tCrB[0][1][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 1]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][2][0][0][0]), as_u32(tCrB[0][2][0][1][0]),
                      as_u32(tCrB[0][2][1][0][0]), as_u32(tCrB[0][2][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 2]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][3][0][0][0]), as_u32(tCrB[0][3][0][1][0]),
                      as_u32(tCrB[0][3][1][0][0]), as_u32(tCrB[0][3][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 3]);
  }
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cm_n = 0; cm_n < CoreMatrix_N; ++cm_n) {
#pragma unroll
        for (int cm_m = 0; cm_m < CoreMatrix_M; ++cm_m) {
          tCrC[m][n][cm_n][cm_m][0] = 0.0f;
          tCrC[m][n][cm_n][cm_m][1] = 0.0f;
        }
      }
    }
  }

#pragma unroll 1
  for (; k_tiles_to_compute > 1; --k_tiles_to_compute) {
    run_mma_tile_n256<K_PIPE_MAX - 2, true>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, StrideA, StrideB, tA_row, tA_col, tB_row, tB_col, warp_m_id,
        warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row, ldsmx4T_col);
  }

#pragma unroll
  for (; k_tiles_to_compute > -(K_PIPE_MAX - 1); --k_tiles_to_compute) {
    run_mma_tile_n256<K_PIPE_MAX - 2, false>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, StrideA, StrideB, tA_row, tA_col, tB_row, tB_col, warp_m_id,
        warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row, ldsmx4T_col);
  }

  cp_async::wait_all();
  __syncthreads();
  float *sC = reinterpret_cast<float *>(shared_memory);

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 264;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][0][0], tCrC[m][n][0][0][1]);
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][1][0], tCrC[m][n][0][1][1]);
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][0][0], tCrC[m][n][1][0][1]);
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][1][0], tCrC[m][n][1][1][1]);
    }
  }

  __syncthreads();

  constexpr int kEpilogueThreads = 256;
  constexpr int kEpilogueElementsPerAccess = 4;
  constexpr int kEpilogueVecCount =
      kCtaM * kCtaN / kEpilogueElementsPerAccess;
  static_assert(kEpilogueVecCount % kEpilogueThreads == 0,
                "epilogue store schedule assumes full fixed-thread coverage");
  constexpr int kEpilogueStoreIterations =
      kEpilogueVecCount / kEpilogueThreads;
  store_gmem_f32_strided<kEpilogueStoreIterations, kCtaN, kEpilogueThreads,
                         kSmemStrideC>(
      gPartial, sC, N);
}



constexpr int kSplitKStages = 3;
constexpr int kSplitKThreads = 256;
constexpr int kSplitKSharedStorageBytes128x256 =
    sizeof(HgemmSharedStorage<shape_mnk_n256, kSplitKStages>);

template <int BlockSwizzle>
inline cudaError_t configure_hgemm_128x256_splitk_fp32acc() {
  auto kernel_fptr =
      sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_kernel<
          shape_mnk_n256, kSplitKStages, BlockSwizzle>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSplitKSharedStorageBytes128x256);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaFuncSetAttribute(kernel_fptr,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              100);
}

inline cudaError_t configure_hgemm_128x256_splitk_fp32acc(int block_swizzle) {
  switch (block_swizzle) {
  case 1:
    return configure_hgemm_128x256_splitk_fp32acc<1>();
  case 2:
    return configure_hgemm_128x256_splitk_fp32acc<2>();
  case 4:
    return configure_hgemm_128x256_splitk_fp32acc<4>();
  case 8:
    return configure_hgemm_128x256_splitk_fp32acc<8>();
  case 16:
    return configure_hgemm_128x256_splitk_fp32acc<16>();
  case 32:
    return configure_hgemm_128x256_splitk_fp32acc<32>();
  case 64:
    return configure_hgemm_128x256_splitk_fp32acc<64>();
  default:
    return cudaErrorInvalidValue;
  }
}

template <int BlockSwizzle>
inline void launch_hgemm_128x256_splitk_fp32acc_unchecked(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k, cudaStream_t stream = 0) {
  int tile_m_count = M / shape_mnk_n256::M;
  int tile_n_count = N / shape_mnk_n256::N;
  dim3 block(kSplitKThreads);
  dim3 grid(tile_m_count * BlockSwizzle,
            (tile_n_count + BlockSwizzle - 1) / BlockSwizzle, split_k);
  sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_kernel<
      shape_mnk_n256, kSplitKStages, BlockSwizzle>
      <<<grid, block, kSplitKSharedStorageBytes128x256, stream>>>(
          A, B, partial, M, N, K, split_k);
}

inline void launch_hgemm_128x256_splitk_fp32acc_unchecked(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k, int block_swizzle, cudaStream_t stream = 0) {
  switch (block_swizzle) {
  case 1:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<1>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 2:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<2>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 4:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<4>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 8:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<8>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 16:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<16>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 32:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<32>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 64:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<64>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  default:
    return;
  }
}

inline cudaError_t launch_hgemm_128x256_splitk_fp32acc(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k, int block_swizzle, cudaStream_t stream = 0) {
  cudaError_t err =
      configure_hgemm_128x256_splitk_fp32acc(block_swizzle);
  if (err != cudaSuccess) {
    return err;
  }
  launch_hgemm_128x256_splitk_fp32acc_unchecked(
      A, B, partial, M, N, K, split_k, block_swizzle, stream);
  return cudaGetLastError();
}

template <int SplitK>
__global__ void sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_reduce_kernel(
    const float *__restrict__ partial, half *__restrict__ C, int M, int N) {
  static_assert(SplitK > 0, "SplitK must be positive");
  constexpr int kTileM = shape_mnk_n256::M;
  constexpr int kTileN = shape_mnk_n256::N;
  constexpr int kThreads = 256;
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerRow = kTileN / kElementsPerVector;
  constexpr int kTileVectors = kTileM * kVectorsPerRow;

  int const tile_m = static_cast<int>(blockIdx.x);
  int const tile_n = static_cast<int>(blockIdx.y);
  int const tid = static_cast<int>(threadIdx.x);
  long long const matrix_elements = static_cast<long long>(M) * N;
  long long const tile_row_base =
      static_cast<long long>(tile_m) * kTileM;
  long long const tile_col_base =
      static_cast<long long>(tile_n) * kTileN;

  for (int vec = tid; vec < kTileVectors; vec += kThreads) {
    int const row = vec / kVectorsPerRow;
    int const col = (vec % kVectorsPerRow) * kElementsPerVector;
    long long const output_index =
        (tile_row_base + row) * N + tile_col_base + col;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma unroll
    for (int split = 0; split < SplitK; ++split) {
      float4 value = *reinterpret_cast<const float4 *>(
          partial + static_cast<long long>(split) * matrix_elements +
          output_index);
      acc0 += value.x;
      acc1 += value.y;
      acc2 += value.z;
      acc3 += value.w;
    }

    half2 out01 = __floats2half2_rn(acc0, acc1);
    half2 out23 = __floats2half2_rn(acc2, acc3);
    half *out = C + output_index;
    *reinterpret_cast<half2 *>(out + 0) = out01;
    *reinterpret_cast<half2 *>(out + 2) = out23;
  }
}

template <int SplitK>
inline cudaError_t launch_hgemm_128x256_splitk_reduce_unchecked(
    const float *partial, half *C, int M, int N,
    cudaStream_t stream = 0) {
  dim3 block(256);
  dim3 grid(M / shape_mnk_n256::M, N / shape_mnk_n256::N);
  sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_reduce_kernel<SplitK>
      <<<grid, block, 0, stream>>>(partial, C, M, N);
  return cudaGetLastError();
}

inline cudaError_t launch_hgemm_128x256_splitk_reduce(
    const float *partial, half *C, int M, int N, int split_k,
    cudaStream_t stream = 0) {
  switch (split_k) {
  case 1:
    return launch_hgemm_128x256_splitk_reduce_unchecked<1>(
        partial, C, M, N, stream);
  case 2:
    return launch_hgemm_128x256_splitk_reduce_unchecked<2>(
        partial, C, M, N, stream);
  case 4:
    return launch_hgemm_128x256_splitk_reduce_unchecked<4>(
        partial, C, M, N, stream);
  case 8:
    return launch_hgemm_128x256_splitk_reduce_unchecked<8>(
        partial, C, M, N, stream);
  case 16:
    return launch_hgemm_128x256_splitk_reduce_unchecked<16>(
        partial, C, M, N, stream);
  case 32:
    return launch_hgemm_128x256_splitk_reduce_unchecked<32>(
        partial, C, M, N, stream);
  case 64:
    return launch_hgemm_128x256_splitk_reduce_unchecked<64>(
        partial, C, M, N, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

} // namespace cuda_ops_core::detail::sm80::splitk
