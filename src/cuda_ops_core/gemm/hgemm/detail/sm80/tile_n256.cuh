#pragma once

#include "tile.cuh"

namespace cuda_ops_core::detail::sm80::tile_n256 {

using namespace ::cuda_ops_core::detail::sm80::common;

__device__ __forceinline__ int offset_B(int n, int k) {
  int n_vec = (n >> 3) ^ (k & 7);
  return (k << 8) + (n_vec << 3) + (n & 7);
}

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_A(half *smem_A, const half *gA,
                                                 int tA_row, int tA_col,
                                                 int strideA) {
  constexpr int kElementsPerAccess = 8;
  int row = tA_row + RowBlock * 32;
  int col = tA_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_A[hgemm_smem::offset_A(row, col)],
                   &gA[row * strideA + col]);
}

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_B(half *smem_B, const half *gB,
                                                 int tB_row, int tB_col,
                                                 int strideB) {
  constexpr int kElementsPerAccess = 8;
  int row = tB_row + RowBlock * 8;
  int col = tB_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_B[offset_B(col, row)], &gB[row * strideB + col]);
}

__device__ __forceinline__ void issue_cp_async_A4(half *smem_A, const half *gA,
                                                  int tA_row, int tA_col,
                                                  int strideA) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kSmemRowStep = 32 * 64;
  int col = tA_col * kElementsPerAccess;
  int smem_base = hgemm_smem::offset_A(tA_row, col);
  const char *gA_bytes = reinterpret_cast<const char *>(gA);
  int gmem_base_bytes = (tA_row * strideA + col) * sizeof(half);
  int gmem_row_step_bytes = 32 * strideA * sizeof(half);
  const char *gA0 = gA_bytes + gmem_base_bytes;
  const char *gA1 = gA0 + gmem_row_step_bytes;
  const char *gA2 = gA1 + gmem_row_step_bytes;
  const char *gA3 = gA2 + gmem_row_step_bytes;

  cp_async::cg<16>(&smem_A[smem_base + 0 * kSmemRowStep], gA0);
  cp_async::cg<16>(&smem_A[smem_base + 1 * kSmemRowStep], gA1);
  cp_async::cg<16>(&smem_A[smem_base + 2 * kSmemRowStep], gA2);
  cp_async::cg<16>(&smem_A[smem_base + 3 * kSmemRowStep], gA3);
}

__device__ __forceinline__ void issue_cp_async_B8(half *smem_B, const half *gB,
                                                  int tB_row, int tB_col,
                                                  int strideB) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kSmemRowStep = 8 * 256;
  int col = tB_col * kElementsPerAccess;
  int smem_base = offset_B(col, tB_row);
  const char *gB_bytes = reinterpret_cast<const char *>(gB);
  int gmem_base_bytes = (tB_row * strideB + col) * sizeof(half);
  int gmem_row_step_bytes = 8 * strideB * sizeof(half);

  cp_async::cg<16>(&smem_B[smem_base + 0 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 0 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 1 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 1 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 2 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 2 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 3 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 3 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 4 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 4 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 5 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 5 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 6 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 6 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 7 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 7 * gmem_row_step_bytes);
}

} // namespace cuda_ops_core::detail::sm80::tile_n256
