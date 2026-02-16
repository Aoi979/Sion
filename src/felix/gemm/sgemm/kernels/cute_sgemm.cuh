#include <cute/tensor.hpp>

template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class AThreadLayout, class TB, class BStride,
          class BSmemLayout, class BThreadLayout, class TC, class CStride,
          class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    CThreadLayout{}))::value) void gemm_device(ProblemShape shape_MNK,
                                               CtaTiler cta_tiler, TA const *A,
                                               AStride dA,
                                               ASmemLayout sA_layout,
                                               AThreadLayout tA, TB const *B,
                                               BStride dB,
                                               BSmemLayout sB_layout,
                                               BThreadLayout tB, TC *C,
                                               CStride dC, CSmemLayout,
                                               CThreadLayout tC, Alpha alpha,
                                               Beta beta) {
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(tC) == size(tA)); // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) ==
                       Int<0>{}); // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) ==
                       Int<0>{}); // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) ==
                       Int<0>{}); // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) ==
                       Int<0>{}); // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) ==
                       Int<0>{}); // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) ==
                       Int<0>{}); // BLK_N / THR_N
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

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K

  // Partition sA (BLK_M, BLK_K) by the rows of tC
  Tensor tCsA =
      local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M,BLK_K)
  // Partition sB (BLK_N, BLK_K) by the cols of tC
  Tensor tCsB =
      local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC =
      local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC); // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB)); // BLK_K

  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    gemm(tCsA, tCsB, tCrC);
    __syncthreads();
  }
  
  axpby(alpha, tCrC, beta, tCgC);
}
