#pragma once

#include "mma.cuh"

namespace cuda_ops_core::detail::sm80::tile {

using namespace ::cuda_ops_core::detail::sm80::common;

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_A(half *smem_A, const half *gA,
                                                 int tA_row, int tA_col,
                                                 int strideA) {
  constexpr int kElementsPerAccess = 8;
  int row = tA_row + RowBlock * 16;
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
  cp_async::cg<16>(&smem_B[hgemm_smem::offset_B(col, row)],
                   &gB[row * strideB + col]);
}

template <typename Shape_MNK> struct Buffer {
  half A[Shape_MNK::M * Shape_MNK::K];
  half B[Shape_MNK::K * Shape_MNK::N];
};
template <typename Shape_MNK, int Stages> struct HgemmSharedStorage {
  Buffer<Shape_MNK> buffer[Stages];
};

struct shape_mnk {
  static constexpr int M = 128;
  static constexpr int N = 128;
  static constexpr int K = 64;
};

struct shape_mnk_n256 {
  static constexpr int M = 128;
  static constexpr int N = 256;
  static constexpr int K = 64;
};


} // namespace cuda_ops_core::detail::sm80::tile


