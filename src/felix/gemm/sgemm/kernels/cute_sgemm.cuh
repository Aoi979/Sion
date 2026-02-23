#include <cute/tensor.hpp>

template <class ProblemShape, class CtaTiler, class AStride, class ASmemLayout,
          class AThreadLayout, class BStride, class BSmemLayout,
          class BThreadLayout, class CStride, class CSmemLayout,
          class CThreadLayout>
__global__ static __launch_bounds__(decltype(size(
    CThreadLayout{}))::value) void gemm_device(ProblemShape shape_MNK,
                                               CtaTiler cta_tiler,
                                               float const *__restrict__ A,
                                               AStride dA,
                                               ASmemLayout sA_layout,
                                               AThreadLayout tA,
                                               float const *__restrict__ B,
                                               BStride dB,
                                               BSmemLayout sB_layout,
                                               BThreadLayout tB,
                                               float *__restrict__ C,
                                               CStride dC,
                                               CSmemLayout sC_layout,
                                               CThreadLayout tC, float alpha,
                                               float beta) {
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

  __shared__ float smemA[cosize_v<ASmemLayout>];
  __shared__ float smemB[cosize_v<BSmemLayout>];
  // TODO: reuse smemA and smemB for sC to reduce shared memory usage
  extern __shared__ float smemC[];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)
  Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout); // (BLK_M,BLK_N)

  Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)
  Tensor tArA = make_fragment_like(tAsA);

  Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)
  Tensor tBrB = make_fragment_like(tBsB);

  Tensor tCgC = local_partition(gC, tC, threadIdx.x); // (THR_M,THR_N)
  Tensor tCsC = local_partition(sC, tC, threadIdx.x); // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K

  // Copy gmem to rmem for k_tile=0
  copy(tAgA(_, _, 0), tArA);
  copy(tBgB(_, _, 0), tBrB);

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

  auto rC = make_tensor_like(glC);

  Tensor swA = local_tile(sA, warp_tiler, warp_coord, Step<_1, X>{});
  Tensor swIterA =
      local_tile(swA, iter_tiler, iter_coord, Step<_1, X>{}); // ((_16),_2,_8)
  Tensor slA = local_tile(swIterA, lane_tiler, lane_coord,
                          Step<_1, X>{}); // ((_4),_2,_8)
  Tensor rA = make_tensor_like(slA);

  Tensor swB = local_tile(sB, warp_tiler, warp_coord, Step<X, _1>{});
  Tensor swIterB =
      local_tile(swB, iter_tiler, iter_coord, Step<X, _1>{}); // ((_16),_2,_8)
  Tensor slB = local_tile(swIterB, lane_tiler, lane_coord,
                          Step<X, _1>{}); // ((_4),_2,_8)

  Tensor rB = make_tensor_like(slB);

  copy(tArA, tAsA);
  copy(tBrB, tBsB);
  __syncthreads();
  copy(slA(_, _, 0), rA(_, _, 0));
  copy(slB(_, _, 0), rB(_, _, 0));
  auto K_TILE_MAX = size<2>(tAgA);
  auto K_BLOCK_MAX = size<2>(rA);

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {
      if (k_block == K_BLOCK_MAX - 1) {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();
      }
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      copy(slA(_, _, k_block_next), rA(_, _, k_block_next));
      copy(slB(_, _, k_block_next), rB(_, _, k_block_next));
      if (k_block == 0) {
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(tAgA(_, _, k_tile_next), tArA);
        copy(tBgB(_, _, k_tile_next), tBrB);
      }
      CUTE_UNROLL
      for (int m = 0; m < 2; ++m) {
        CUTE_UNROLL
        for (int n = 0; n < 2; ++n) {
          gemm(rA(_, m, k_block), rB(_, n, k_block), rC(_, _, m, n));
        }
      }
    }
  }
  copy(rC, slC);
  __syncthreads();
  axpby(alpha, tCsC, beta, tCgC);
}
