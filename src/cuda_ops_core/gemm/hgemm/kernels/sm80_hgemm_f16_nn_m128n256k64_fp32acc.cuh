#pragma once

#include "../detail/sm80/mma_macros.cuh"
#include "../detail/sm80/tile.cuh"
#include "../detail/sm80/tile_n256.cuh"

namespace cuda_ops_core::detail::sm80::fp32acc {
using namespace ::cuda_ops_core::detail::sm80::common;
using namespace ::cuda_ops_core::detail::sm80::tile;
using namespace ::cuda_ops_core::detail::sm80::tile_n256;

namespace m128n256 {

template <int WaitGroup, bool AdvanceGmem, typename Shape_MNK, int kStages>
__device__ __forceinline__ void run_mma_tile_n256(
    float (&tCrC)[4][4][2][2][2],
    half (&tCrA)[2][4][2][2][2],
    half (&tCrB)[2][4][2][2][2],
    HgemmSharedStorage<Shape_MNK, kStages> *smem, int &smem_read_offset,
    int &smem_write_offset, const half *&gA_next, const half *&gB_next,
    int StrideA, int StrideB,
    int tA_row, int tA_col, int tB_row, int tB_col, int warp_m_id,
    int warp_n_id, int ldsmx4_row, int ldsmx4_col, int ldsmx4T_row,
    int ldsmx4T_col) {
  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 64;
  constexpr int Tiled_MMA_K = 16;
  constexpr int K_BLOCK_MAX = 4;
  constexpr int kBufferBytes = sizeof(Buffer<Shape_MNK>);
  constexpr int kAElements = Shape_MNK::M * Shape_MNK::K;
  static_assert(WaitGroup == 0 || WaitGroup == 1,
                "n256 pipeline only uses wait_group 0 or 1");

  char *smem_bytes = reinterpret_cast<char *>(smem);
  half *smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
  half *smem_read_B = smem_read_A + kAElements;
  half *smem_write_A =
      reinterpret_cast<half *>(smem_bytes + smem_write_offset);
  half *smem_write_B = smem_write_A + kAElements;

  #pragma unroll
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    int k_block_next = (k_block + 1) % K_BLOCK_MAX;
    int k_block_slot = k_block & 1;
    int k_block_next_slot = k_block_next & 1;
    half *b_smem = smem_read_B;
    int b_ldsm_base = tile_n256::offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32,
        ldsmx4T_col + k_block_next * Tiled_MMA_K);
    if (k_block == 0) {
      tile_n256::issue_cp_async_A4(smem_write_A, gA_next, tA_row, tA_col,
                                   StrideA);
      tile_n256::issue_cp_async_B8(smem_write_B, gB_next, tB_row, tB_col,
                                   StrideB);
      if constexpr (AdvanceGmem) {
        gA_next += Shape_MNK::K;
        gB_next += Shape_MNK::K * StrideB;
      }
    }
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][0][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][0][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][0][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][0][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][0][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][0][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][0][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][0][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 0]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][1][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][1][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][1][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][1][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 1]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][2][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][2][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][2][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][2][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 2]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][3][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][3][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][3][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][3][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 3]);

    MMA_1_ROW_SLOT_FP32(0, k_block_slot);
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][1][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][1][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][1][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][1][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    MMA_1_ROW_SLOT_FP32(1, k_block_slot);
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][2][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][2][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][2][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][2][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    MMA_1_ROW_SLOT_FP32(2, k_block_slot);
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][3][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][3][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][3][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][3][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    MMA_1_ROW_SLOT_FP32(3, k_block_slot);
    if (k_block == K_BLOCK_MAX - 2) {
      cp_async::commit_group();
      smem_write_offset = smem_read_offset;
      smem_read_offset =
          (smem_read_offset == (kStages - 1) * kBufferBytes)
              ? 0
              : smem_read_offset + kBufferBytes;
      smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
      smem_read_B = smem_read_A + kAElements;
      smem_write_A = reinterpret_cast<half *>(smem_bytes + smem_write_offset);
      smem_write_B = smem_write_A + kAElements;
      cp_async::wait_group<WaitGroup>();
      __syncthreads();
    }
  }
}

template <typename Shape_MNK = shape_mnk_n256, int kStages, int kBlockSwizzle>
__global__ void sm80_hgemm_f16_nn_m128n256k64_fp32acc_kernel(
    half *A, half *B, half *C, int M, int N, int K) {
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
  int StrideC = N;

  int const tile_m_max = (M + kCtaM - 1) / kCtaM;
  int const tile_n_max = (N + kCtaN - 1) / kCtaN;

  int tile_m = blockIdx.x / kBlockSwizzle;
  int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tile_m >= tile_m_max || tile_n >= tile_n_max) {
    return;
  }

  const half *gA_base = A + tile_m * kCtaM * StrideA;
  const half *gB_base = B + tile_n * kCtaN;

  half *gC = C + tile_m * kCtaM * StrideC + tile_n * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;

  int const K_TILE_MAX = K / kCtaK;
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
    tile_n256::issue_cp_async_A4(smem->buffer[k_pipe].A, gA_next, tA_row,
                                 tA_col, StrideA);
    tile_n256::issue_cp_async_B8(smem->buffer[k_pipe].B, gB_next, tB_row,
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
  half *sC = reinterpret_cast<half *>(shared_memory);

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 264;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][0][0][0], tCrC[m][n][0][0][1]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][0][1][0], tCrC[m][n][0][1][1]);

      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][1][0][0], tCrC[m][n][1][0][1]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][1][1][0], tCrC[m][n][1][1][1]);
    }
  }

  __syncthreads();

  constexpr int kEpilogueThreads = 256;
  constexpr int kEpilogueVecCount = kCtaM * kCtaN / kElementsPerAccess;
  static_assert(kEpilogueVecCount % kEpilogueThreads == 0,
                "epilogue store schedule assumes full fixed-thread coverage");
  constexpr int kEpilogueStoreIterations = kEpilogueVecCount / kEpilogueThreads;
  hgemm_epilogue::store_gmem_strided<kEpilogueStoreIterations, kCtaN,
                                     kElementsPerAccess, kEpilogueThreads,
                                     kSmemStrideC>(gC, sC, StrideC);
}

} // namespace m128n256
} // namespace cuda_ops_core::detail::sm80::fp32acc
