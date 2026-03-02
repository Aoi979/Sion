#include <cute/tensor.hpp>

template <class ProblemShape, class CtaTiler, class AStride, class ASmemLayout,
          class AThreadLayout, class BStride, class BSmemLayout,
          class BThreadLayout, class CStride, class CSmemLayout,
          class CThreadLayout>
__global__ static __launch_bounds__(
    decltype(size(CThreadLayout{}))::
        value) void gemm_device_wt_db(ProblemShape shape_MNK,
                                      CtaTiler cta_tiler,
                                      float const *__restrict__ A, AStride dA,
                                      ASmemLayout sA_layout, AThreadLayout tA,
                                      float const *__restrict__ B, BStride dB,
                                      BSmemLayout sB_layout, BThreadLayout tB,
                                      float *__restrict__ C, CStride dC,
                                      CSmemLayout sC_layout, CThreadLayout tC,
                                      float alpha, float beta) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB)); // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) ==
                       Int<0>{}); // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) ==
                       Int<0>{}); // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) ==
                       Int<0>{}); // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) ==
                       Int<0>{}); // BLK_K / THR_K

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(
      congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

  Tensor mA =
      make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB =
      make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC =
      make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                         Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                         Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  __shared__ float smem[cosize_v<ASmemLayout> + cosize_v<BSmemLayout>];
  auto smemA = smem;
  auto smemB = smemA + cosize_v<ASmemLayout>;
  auto smemC = smem;
  Tensor sA =
      make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K, PIPE)
  Tensor sB =
      make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K, PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout); // (BLK_M,BLK_N)

  Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K, PIPE)
  Tensor tArA = make_fragment_like(tAsA(_, _, 0));

  Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K, PIPE)
  Tensor tBrB = make_fragment_like(tBsB(_, _, 0));

  // Tensor tCgC = local_partition(gC, tC, threadIdx.x); // (THR_M,THR_N)
  // Tensor tCsC = local_partition(sC, tC, threadIdx.x); // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K

  auto warp_id = threadIdx.x / 32;
  auto lane_id = threadIdx.x % 32;

  auto warp_tiler = make_shape(Int<32>{}, Int<64>{});
  auto iter_tiler = make_shape(Int<16>{}, Int<32>{});
  auto lane_tiler = make_shape(Int<4>{}, Int<4>{});
  auto iter_coord = make_coord(_, _);
  auto lane_coord = make_coord(lane_id / 8, lane_id % 8);
  auto warp_coord = make_coord(warp_id / 2, warp_id % 2);

  Tensor gwC = local_tile(gC, warp_tiler, warp_coord);      //(32, 64)
  Tensor swC = local_tile(sC, warp_tiler, warp_coord);      //(32, 64)
  Tensor gwIterC = local_tile(gwC, iter_tiler, iter_coord); //(16, 32, 2, 2)
  Tensor swIterC = local_tile(swC, iter_tiler, iter_coord); //(16, 32, 2, 2)
  Tensor glC = local_tile(gwIterC, lane_tiler, lane_coord); //(4, 4, 2, 2)
  Tensor slC = local_tile(swIterC, lane_tiler, lane_coord); //(4, 4, 2, 2)

  Tensor rC = make_tensor_like(glC);

  Tensor swA = local_tile(sA, warp_tiler, warp_coord, Step<_1, X>{});
  Tensor swIterA = local_tile(swA, iter_tiler, iter_coord,
                              Step<_1, X>{}); // ((_16),_2,_8, PIPE)
  Tensor slA = local_tile(swIterA, lane_tiler, lane_coord,
                          Step<_1, X>{}); // ((_4),_2,_8, PIPE)
  Tensor rA = make_tensor_like(slA(_, _, _, 0));

  Tensor swB = local_tile(sB, warp_tiler, warp_coord, Step<X, _1>{});
  Tensor swIterB = local_tile(swB, iter_tiler, iter_coord,
                              Step<X, _1>{}); // ((_16),_2,_8, PIPE)
  Tensor slB = local_tile(swIterB, lane_tiler, lane_coord,
                          Step<X, _1>{}); // ((_4),_2,_8, PIPE)

  Tensor rB = make_tensor_like(slB(_, _, _, 0));

  // Copy gmem to rmem for k_tile=0
  copy(tAgA(_, _, 0), tArA);
  copy(tBgB(_, _, 0), tBrB);
  // rmem -> smem
  copy(tArA, tAsA(_, _, 0));
  copy(tBrB, tBsB(_, _, 0));
  __syncthreads();
  // bank conflict
  copy(slA(_, _, 0, 0), rA(_, _, 0));
  copy(slB(_, _, 0, 0), rB(_, _, 0));

  uint32_t write_stage_idx = 1;
  auto K_TILE_MAX = size<2>(tAgA);
  auto K_BLOCK_MAX = size<2>(rA);

  CUTE_NO_UNROLL
  for (int k_tile = 1; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tAgA(_, _, k_tile), tArA);
    copy(tBgB(_, _, k_tile), tBrB);
    uint32_t load_stage_idx = write_stage_idx ^ 1;
    CUTE_UNROLL
    for (int k_block = 1; k_block < K_BLOCK_MAX; k_block++) {
      copy(slA(_, _, k_block, load_stage_idx), rA(_, _, k_block));
      copy(slB(_, _, k_block, load_stage_idx), rB(_, _, k_block));
      CUTE_UNROLL
      for (int m = 0; m < 2; ++m) {
        CUTE_UNROLL
        for (int n = 0; n < 2; ++n) {
          gemm(rA(_, m, k_block - 1), rB(_, n, k_block - 1), rC(_, _, m, n));
        }
      }
    }

    copy(tArA, tAsA(_, _, write_stage_idx));
    copy(tBrB, tBsB(_, _, write_stage_idx));
    __syncthreads();
    // bank conflict
    copy(slA(_, _, 0, write_stage_idx), rA(_, _, 0));
    copy(slB(_, _, 0, write_stage_idx), rB(_, _, 0));
    write_stage_idx ^= 1;
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < 2; ++n) {
        gemm(rA(_, m, K_BLOCK_MAX - 1), rB(_, n, K_BLOCK_MAX - 1),
             rC(_, _, m, n));
      }
    }
  }
  uint32_t load_stage_idx = (K_TILE_MAX - 1) % 2;
  CUTE_UNROLL
  for (int k_block = 1; k_block < K_BLOCK_MAX; k_block++) {
    copy(slA(_, _, k_block, load_stage_idx), rA(_, _, k_block));
    copy(slB(_, _, k_block, load_stage_idx), rB(_, _, k_block));
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < 2; ++n) {
        gemm(rA(_, m, k_block - 1), rB(_, n, k_block - 1), rC(_, _, m, n));
      }
    }
  }
  CUTE_UNROLL
  for (int m = 0; m < 2; ++m) {
    CUTE_UNROLL
    for (int n = 0; n < 2; ++n) {
      gemm(rA(_, m, K_BLOCK_MAX - 1), rB(_, n, K_BLOCK_MAX - 1),
           rC(_, _, m, n));
    }
  }
  axpby(alpha, rC, beta, glC);
}
