#pragma once

#include "mma_macros.cuh"
#include "mma.cuh"
#include "tile.cuh"
#include "tile_n256.cuh"

namespace cuda_ops_core::detail::sm80::splitk {

using namespace ::cuda_ops_core::detail::sm80::common;
using namespace ::cuda_ops_core::detail::sm80::tile;
using namespace ::cuda_ops_core::detail::sm80::tile_n256;

namespace support {

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
    int b_ldsm_base = offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32,
        ldsmx4T_col + k_block_next * Tiled_MMA_K);
    if (k_block == 0) {
      issue_cp_async_A4(smem_write_A, gA_next, tA_row, tA_col, StrideA);
      issue_cp_async_B8(smem_write_B, gB_next, tB_row, tB_col, StrideB);
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


__device__ __forceinline__ void store_f32x2(float *sC, int row, int col,
                                            int stride, float x0, float x1) {
  float *ptr = sC + row * stride + col;
  ptr[0] = x0;
  ptr[1] = x1;
}

template <int kStoreIterations, int kCtaN, int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_f32_strided(float *gC,
                                                       const float *sC,
                                                       int strideC) {
  constexpr int kElementsPerAccess = 4; // four fp32 values, 16B
  constexpr int kVecsPerRow = kCtaN / kElementsPerAccess;
  constexpr int kRowsPerStep = kThreads / kVecsPerRow;

  int vec_row = threadIdx.x / kVecsPerRow;
  int vec_col = threadIdx.x % kVecsPerRow;
  float *d_ptr = gC + vec_row * strideC + vec_col * kElementsPerAccess;
  const float *s_ptr = sC + vec_row * kSmemStrideC +
                      vec_col * kElementsPerAccess;
  int d_step = kRowsPerStep * strideC;
  constexpr int s_step = kRowsPerStep * kSmemStrideC;

#pragma unroll
  for (int i = 0; i < kStoreIterations; ++i) {
    *reinterpret_cast<uint4 *>(d_ptr) =
        *reinterpret_cast<const uint4 *>(s_ptr);
    d_ptr += d_step;
    s_ptr += s_step;
  }
}


} // namespace support

} // namespace cuda_ops_core::detail::sm80::splitk
