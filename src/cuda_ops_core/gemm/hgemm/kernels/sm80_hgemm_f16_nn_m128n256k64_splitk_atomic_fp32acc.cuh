#pragma once

#include "../detail/sm80/splitk.cuh"

namespace cuda_ops_core::detail::sm80::atomic_splitk {

using namespace ::cuda_ops_core::detail::sm80::common;
using namespace ::cuda_ops_core::detail::sm80::tile;
namespace base = ::cuda_ops_core::detail::sm80::splitk::support;
namespace tile = ::cuda_ops_core::detail::sm80::tile_n256;

__device__ __forceinline__ void wait_for_tile_turn(int *tile_turn,
                                                   int split_id) {
  while (atomicAdd(tile_turn, 0) != split_id) {
    __nanosleep(64);
  }
}

__device__ __forceinline__ void atomic_add_f32x2_and_store(
    float *accumulator, half *C, long long index, float x0, float x1,
    bool final_split) {
  float old0 = atomicAdd(accumulator + index + 0, x0);
  float old1 = atomicAdd(accumulator + index + 1, x1);
  if (final_split) {
    *reinterpret_cast<half2 *>(C + index) =
        __floats2half2_rn(old0 + x0, old1 + x1);
  }
}

template <typename Shape_MNK = shape_mnk_n256, int kStages,
          int kBlockSwizzle>
__global__ void
sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc_kernel(
    const half *__restrict__ A, const half *__restrict__ B,
    float *__restrict__ accumulator, int *__restrict__ tile_turns,
    half *__restrict__ C, int M, int N, int K, int split_k) {
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

  int const StrideA = K;
  int const StrideB = N;
  int const tile_m_max = (M + kCtaM - 1) / kCtaM;
  int const tile_n_max = (N + kCtaN - 1) / kCtaN;

  int const tile_m = blockIdx.x / kBlockSwizzle;
  int const tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
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

  long long const tile_output_index =
      static_cast<long long>(tile_m) * kCtaM * N +
      static_cast<long long>(tile_n) * kCtaN;
  float *gAccumulator = accumulator + tile_output_index;
  half *gC = C + tile_output_index;

  long long const tile_index =
      static_cast<long long>(tile_m) * tile_n_max + tile_n;
  int *tile_turn = tile_turns + tile_index;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane_id = tid % kWarpSize;

  constexpr int kElementsPerAccess = 8; // half, 16B
  int tA_row = tid / (kCtaK / kElementsPerAccess); // 8
  int tA_col = tid % (kCtaK / kElementsPerAccess);
  int tB_row = tid / (kCtaN / kElementsPerAccess); // 32
  int tB_col = tid % (kCtaN / kElementsPerAccess);

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

  float tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][Fragment];
  half tCrA[kFragmentSlots][MMA_M][CoreMatrix_K][CoreMatrix_M][Fragment];
  half tCrB[kFragmentSlots][MMA_N][CoreMatrix_N][CoreMatrix_K][Fragment];

  const half *gA_next = gA_base;
  const half *gB_next = gB_base;
  int k_tiles_to_compute = k_tiles_per_split;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    tile::issue_cp_async_A4(smem->buffer[k_pipe].A, gA_next, tA_row,
                                   tA_col, StrideA);
    tile::issue_cp_async_B8(smem->buffer[k_pipe].B, gB_next, tB_row,
                                   tB_col, StrideB);

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

    ldsm::x4<ldsm::N>(
        as_u32(tCrA[0][0][0][0][0]), as_u32(tCrA[0][0][0][1][0]),
        as_u32(tCrA[0][0][1][0][0]), as_u32(tCrA[0][0][1][1][0]),
        smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(
        as_u32(tCrA[0][1][0][0][0]), as_u32(tCrA[0][1][0][1][0]),
        as_u32(tCrA[0][1][1][0][0]), as_u32(tCrA[0][1][1][1][0]),
        smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(
        as_u32(tCrA[0][2][0][0][0]), as_u32(tCrA[0][2][0][1][0]),
        as_u32(tCrA[0][2][1][0][0]), as_u32(tCrA[0][2][1][1][0]),
        smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(
        as_u32(tCrA[0][3][0][0][0]), as_u32(tCrA[0][3][0][1][0]),
        as_u32(tCrA[0][3][1][0][0]), as_u32(tCrA[0][3][1][1][0]),
        smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));

    half *b_smem = smem_read_B;
    int b_ldsm_base = tile::offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32, ldsmx4T_col);
    ldsm::x4<ldsm::T>(
        as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
        as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
        &b_smem[b_ldsm_base + Tiled_MMA_N * 0]);
    ldsm::x4<ldsm::T>(
        as_u32(tCrB[0][1][0][0][0]), as_u32(tCrB[0][1][0][1][0]),
        as_u32(tCrB[0][1][1][0][0]), as_u32(tCrB[0][1][1][1][0]),
        &b_smem[b_ldsm_base + Tiled_MMA_N * 1]);
    ldsm::x4<ldsm::T>(
        as_u32(tCrB[0][2][0][0][0]), as_u32(tCrB[0][2][0][1][0]),
        as_u32(tCrB[0][2][1][0][0]), as_u32(tCrB[0][2][1][1][0]),
        &b_smem[b_ldsm_base + Tiled_MMA_N * 2]);
    ldsm::x4<ldsm::T>(
        as_u32(tCrB[0][3][0][0][0]), as_u32(tCrB[0][3][0][1][0]),
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
    base::run_mma_tile_n256<K_PIPE_MAX - 2, true>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset,
        gA_next, gB_next, StrideA, StrideB, tA_row, tA_col, tB_row, tB_col,
        warp_m_id, warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row,
        ldsmx4T_col);
  }

#pragma unroll
  for (; k_tiles_to_compute > -(K_PIPE_MAX - 1); --k_tiles_to_compute) {
    base::run_mma_tile_n256<K_PIPE_MAX - 2, false>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset,
        gA_next, gB_next, StrideA, StrideB, tA_row, tA_col, tB_row, tB_col,
        warp_m_id, warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row,
        ldsmx4T_col);
  }

  cp_async::wait_all();
  __syncthreads();

  wait_for_tile_turn(tile_turn, split_id);
  bool const final_split = split_id + 1 == split_k;
  int const core_matrix_row = lane_id / 4;
  int const core_matrix_col = lane_id % 4;

#define ATOMIC_STORE_F32X2(MM, NN, CMN, CMM, ROW, COL)                       \
  atomic_add_f32x2_and_store(                                                  \
      gAccumulator, gC,                                                        \
      static_cast<long long>(ROW) * N + (COL),                                 \
      tCrC[MM][NN][CMN][CMM][0], tCrC[MM][NN][CMN][CMM][1], final_split)

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
      int const row0 = m * Tiled_MMA_M + warp_m_id * 16 + core_matrix_row;
      int const col0 = n * Tiled_MMA_N + warp_n_id * 8 + core_matrix_col * 2;
      int const row1 = row0 + 8;
      int const col1 = col0 + 32;
      ATOMIC_STORE_F32X2(m, n, 0, 0, row0, col0);
      ATOMIC_STORE_F32X2(m, n, 0, 1, row1, col0);
      ATOMIC_STORE_F32X2(m, n, 1, 0, row0, col1);
      ATOMIC_STORE_F32X2(m, n, 1, 1, row1, col1);
    }
  }

#undef ATOMIC_STORE_F32X2

  __syncthreads();
  if (tid == 0) {
    __threadfence();
    atomicExch(tile_turn, split_id + 1);
  }
}

constexpr int kAtomicSplitKStages = 3;
constexpr int kAtomicSplitKThreads = 256;
constexpr int kAtomicSplitKSharedStorageBytes128x256 =
    sizeof(HgemmSharedStorage<shape_mnk_n256, kAtomicSplitKStages>);

template <int BlockSwizzle>
inline cudaError_t configure_hgemm_128x256_splitk_atomic_fp32acc() {
  auto kernel_fptr =
      sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc_kernel<
          shape_mnk_n256, kAtomicSplitKStages, BlockSwizzle>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kAtomicSplitKSharedStorageBytes128x256);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaFuncSetAttribute(kernel_fptr,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              100);
}

inline cudaError_t configure_hgemm_128x256_splitk_atomic_fp32acc(
    int block_swizzle) {
  switch (block_swizzle) {
  case 1:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<1>();
  case 2:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<2>();
  case 4:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<4>();
  case 8:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<8>();
  case 16:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<16>();
  case 32:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<32>();
  case 64:
    return configure_hgemm_128x256_splitk_atomic_fp32acc<64>();
  default:
    return cudaErrorInvalidValue;
  }
}

template <int BlockSwizzle>
inline void launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked(
    const half *A, const half *B, float *accumulator, int *tile_turns,
    half *C, int M, int N, int K, int split_k, cudaStream_t stream = 0) {
  int const tile_m_count = M / shape_mnk_n256::M;
  int const tile_n_count = N / shape_mnk_n256::N;
  dim3 block(kAtomicSplitKThreads);
  dim3 grid(tile_m_count * BlockSwizzle,
            (tile_n_count + BlockSwizzle - 1) / BlockSwizzle, split_k);
  sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc_kernel<
      shape_mnk_n256, kAtomicSplitKStages, BlockSwizzle>
      <<<grid, block, kAtomicSplitKSharedStorageBytes128x256, stream>>>(
          A, B, accumulator, tile_turns, C, M, N, K, split_k);
}

inline void launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked(
    const half *A, const half *B, float *accumulator, int *tile_turns, half *C,
    int M, int N, int K, int split_k, int block_swizzle,
    cudaStream_t stream = 0) {
  switch (block_swizzle) {
  case 1:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<1>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  case 2:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<2>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  case 4:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<4>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  case 8:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<8>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  case 16:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<16>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  case 32:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<32>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  case 64:
    launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked<64>(
        A, B, accumulator, tile_turns, C, M, N, K, split_k, stream);
    return;
  default:
    return;
  }
}

inline cudaError_t launch_hgemm_128x256_splitk_atomic_fp32acc(
    const half *A, const half *B, float *accumulator, int *tile_turns, half *C,
    int M, int N, int K, int split_k, int block_swizzle,
    cudaStream_t stream = 0) {
  cudaError_t err =
      configure_hgemm_128x256_splitk_atomic_fp32acc(block_swizzle);
  if (err != cudaSuccess) {
    return err;
  }
  launch_hgemm_128x256_splitk_atomic_fp32acc_unchecked(
      A, B, accumulator, tile_turns, C, M, N, K, split_k, block_swizzle,
      stream);
  return cudaGetLastError();
}

} // namespace cuda_ops_core::detail::sm80::atomic_splitk
