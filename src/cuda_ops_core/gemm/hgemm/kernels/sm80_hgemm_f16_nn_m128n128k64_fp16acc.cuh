#pragma once

#include "../detail/sm80/mma_macros.cuh"
#include "../detail/sm80/tile.cuh"

namespace cuda_ops_core::detail::sm80::fp16acc {
using namespace ::cuda_ops_core::detail::sm80::common;
using namespace ::cuda_ops_core::detail::sm80::tile;

template <typename Shape_MNK = shape_mnk, int kStages, int kBlockSwizzle>
__global__ void hgemm_f16f16f16_kernel(half *A, half *B, half *C, int M, int N,
                                       int K) {
  constexpr int kCtaM = Shape_MNK::M; // 128
  constexpr int kCtaN = Shape_MNK::N; // 128
  constexpr int kCtaK = Shape_MNK::K; // 64
  static_assert(kCtaM == 128 && kCtaN == 128 && kCtaK == 64,
                "swizzled shared-memory layout assumes a 128x128x64 CTA");

  constexpr int kWarpsM = 2;
  constexpr int kWarpSize = 32;

  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 32;
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
  constexpr int kFragmentSlots = 2;

  constexpr int Fragment = 2;
  constexpr int CoreMatrix_M = 2;
  constexpr int CoreMatrix_N = 2;
  constexpr int CoreMatrix_K = 2;

  constexpr int kElementsPerAccess = 8; // half, 16B

  half tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][Fragment];
  half tCrA[MMA_M][kFragmentSlots][CoreMatrix_K][CoreMatrix_M][Fragment];
  half tCrB[MMA_N][kFragmentSlots][CoreMatrix_N][CoreMatrix_K][Fragment];

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cm_n = 0; cm_n < CoreMatrix_N; ++cm_n) {
#pragma unroll
        for (int cm_m = 0; cm_m < CoreMatrix_M; ++cm_m) {
          as_u32(tCrC[m][n][cm_n][cm_m][0]) = 0;
        }
      }
    }
  }

  int lane_id = tid % kWarpSize;

  int tA_row = tid / (kCtaK / kElementsPerAccess); // 8
  int tA_col = tid % (kCtaK / kElementsPerAccess);

  int tB_row = tid / (kCtaN / kElementsPerAccess); // 16
  int tB_col = tid % (kCtaN / kElementsPerAccess);

  int k_tiles_to_issue = K_TILE_MAX;
  int k_tiles_to_compute = K_TILE_MAX;
  int k_tile_next = 0;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    const half *gA = gA_base + k_tile_next * kCtaK;
    const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 0 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 0 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 1 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 1 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 2 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 2 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 3 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 3 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 4 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 4 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 5 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 5 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 6 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 6 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 7 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 7 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 0 * 8)],
        &gB[(tB_row + 0 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 1 * 8)],
        &gB[(tB_row + 1 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 2 * 8)],
        &gB[(tB_row + 2 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 3 * 8)],
        &gB[(tB_row + 3 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 4 * 8)],
        &gB[(tB_row + 4 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 5 * 8)],
        &gB[(tB_row + 5 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 6 * 8)],
        &gB[(tB_row + 6 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 7 * 8)],
        &gB[(tB_row + 7 * 8) * StrideB + tB_col * kElementsPerAccess]);

    cp_async::commit_group();
    --k_tiles_to_issue;
    ++k_tile_next;
  }

  int smem_pipe_read = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

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
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[1][0][0][0][0]), as_u32(tCrA[1][0][0][1][0]),
                      as_u32(tCrA[1][0][1][0][0]), as_u32(tCrA[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[2][0][0][0][0]), as_u32(tCrA[2][0][0][1][0]),
                      as_u32(tCrA[2][0][1][0][0]), as_u32(tCrA[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[3][0][0][0][0]), as_u32(tCrA[3][0][0][1][0]),
                      as_u32(tCrA[3][0][1][0][0]), as_u32(tCrA[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 0 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[1][0][0][0][0]), as_u32(tCrB[1][0][0][1][0]),
                      as_u32(tCrB[1][0][1][0][0]), as_u32(tCrB[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 1 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[2][0][0][0][0]), as_u32(tCrB[2][0][0][1][0]),
                      as_u32(tCrB[2][0][1][0][0]), as_u32(tCrB[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 2 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[3][0][0][0][0]), as_u32(tCrB[3][0][0][1][0]),
                      as_u32(tCrB[3][0][1][0][0]), as_u32(tCrB[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 3 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
  }
  auto stage_A_p = smem->buffer[smem_pipe_read].A;
  auto stage_B_p = smem->buffer[smem_pipe_read].B;

  while (k_tiles_to_compute > 0) {
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_tiles_to_issue > 0) {
        const half *gA = gA_base + k_tile_next * kCtaK;
        const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
        half *sA = smem->buffer[smem_pipe_write].A;
        half *sB = smem->buffer[smem_pipe_write].B;

        if (k_block == 0) {
          issue_cp_async_A<0>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<1>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<0>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<1>(sB, gB, tB_row, tB_col, StrideB);
        } else if (k_block == 1) {
          issue_cp_async_A<2>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<3>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<2>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<3>(sB, gB, tB_row, tB_col, StrideB);
        } else if (k_block == 2) {
          issue_cp_async_A<4>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<5>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<4>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<5>(sB, gB, tB_row, tB_col, StrideB);
        } else {
          issue_cp_async_A<6>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<7>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<6>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<7>(sB, gB, tB_row, tB_col, StrideB);
        }
      }

      if (k_block == K_BLOCK_MAX - 1) {
        cp_async::commit_group();
        if (k_tiles_to_issue > 0) {
          --k_tiles_to_issue;
          ++k_tile_next;
        }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;

        stage_A_p = smem->buffer[smem_pipe_read].A;
        stage_B_p = smem->buffer[smem_pipe_read].B;
        if (k_tiles_to_compute <= K_PIPE_MAX - 1) {
          cp_async::wait_group<0>();
        } else {
          cp_async::wait_group<K_PIPE_MAX - 2>();
        }
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      int k_block_slot = k_block & 1;
      int k_block_next_slot = k_block_next & 1;
      ldsm::x4<ldsm::N>(as_u32(tCrA[0][k_block_next_slot][0][0][0]),
                        as_u32(tCrA[0][k_block_next_slot][0][1][0]),
                        as_u32(tCrA[0][k_block_next_slot][1][0][0]),
                        as_u32(tCrA[0][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[1][k_block_next_slot][0][0][0]),
                        as_u32(tCrA[1][k_block_next_slot][0][1][0]),
                        as_u32(tCrA[1][k_block_next_slot][1][0][0]),
                        as_u32(tCrA[1][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[2][k_block_next_slot][0][0][0]),
                        as_u32(tCrA[2][k_block_next_slot][0][1][0]),
                        as_u32(tCrA[2][k_block_next_slot][1][0][0]),
                        as_u32(tCrA[2][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[3][k_block_next_slot][0][0][0]),
                        as_u32(tCrA[3][k_block_next_slot][0][1][0]),
                        as_u32(tCrA[3][k_block_next_slot][1][0][0]),
                        as_u32(tCrA[3][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[0][k_block_next_slot][0][0][0]),
                        as_u32(tCrB[0][k_block_next_slot][0][1][0]),
                        as_u32(tCrB[0][k_block_next_slot][1][0][0]),
                        as_u32(tCrB[0][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 0 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[1][k_block_next_slot][0][0][0]),
                        as_u32(tCrB[1][k_block_next_slot][0][1][0]),
                        as_u32(tCrB[1][k_block_next_slot][1][0][0]),
                        as_u32(tCrB[1][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 1 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[2][k_block_next_slot][0][0][0]),
                        as_u32(tCrB[2][k_block_next_slot][0][1][0]),
                        as_u32(tCrB[2][k_block_next_slot][1][0][0]),
                        as_u32(tCrB[2][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 2 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[3][k_block_next_slot][0][0][0]),
                        as_u32(tCrB[3][k_block_next_slot][0][1][0]),
                        as_u32(tCrB[3][k_block_next_slot][1][0][0]),
                        as_u32(tCrB[3][k_block_next_slot][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 3 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);

      MMA_1_ROW_SLOT(0, k_block_slot);
      MMA_1_ROW_SLOT(1, k_block_slot);
      MMA_1_ROW_SLOT(2, k_block_slot);
      MMA_1_ROW_SLOT(3, k_block_slot);
    }
    --k_tiles_to_compute;
  }

  cp_async::wait_all();
  __syncthreads();
  half *sC = reinterpret_cast<half *>(shared_memory);

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 136;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][0][0][0]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][0][1][0]);

      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][1][0][0]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][1][1][0]);
    }
  }

  __syncthreads();

  constexpr int kEpilogueThreads = 128;
  constexpr int kEpilogueVecCount = kCtaM * kCtaN / kElementsPerAccess;
  static_assert(kEpilogueVecCount % kEpilogueThreads == 0,
                "epilogue store schedule assumes full fixed-thread coverage");
  constexpr int kEpilogueStoreIterations = kEpilogueVecCount / kEpilogueThreads;
  hgemm_epilogue::store_gmem_strided<kEpilogueStoreIterations, kCtaN,
                                     kElementsPerAccess, kEpilogueThreads,
                                     kSmemStrideC>(gC, sC, StrideC);
}

} // namespace cuda_ops_core::detail::sm80::fp16acc
