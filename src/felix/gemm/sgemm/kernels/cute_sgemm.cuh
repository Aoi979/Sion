#include "cute/config.hpp"
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
                                               CStride dC, CSmemLayout,
                                               CThreadLayout, float alpha,
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
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

  Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

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

  Tensor wC = local_tile(gC, warp_tiler, warp_coord);     //(32, 64)
  Tensor wIterC = local_tile(wC, iter_tiler, iter_coord); //(16, 32, 2, 2)
  Tensor lC = local_tile(wIterC, lane_tiler, lane_coord); //(4, 4, 2, 2)
  auto rC = make_tensor_like(lC);

  Tensor wA = local_tile(sA, warp_tiler, warp_coord, Step<_1, X>{});
  Tensor wIterA =
      local_tile(wA, iter_tiler, iter_coord, Step<_1, X>{}); // ((_16),_2,_16)
  Tensor lA = local_tile(wIterA, lane_tiler, lane_coord,
                         Step<_1, X>{}); // ((_4),_2,_16)

  Tensor wB = local_tile(sB, warp_tiler, warp_coord, Step<X, _1>{});
  Tensor wIterB =
      local_tile(wB, iter_tiler, iter_coord, Step<X, _1>{}); // ((_16),_2,_16)
  Tensor lB = local_tile(wIterB, lane_tiler, lane_coord,
                         Step<X, _1>{}); // ((_4),_2,_16)
  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tAgA(_, _, k_tile), tAsA); // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_, _, k_tile), tBsB); // B   (THR_N,THR_K) -> (THR_N,THR_K)

    cp_async_fence();
    cp_async_wait<0>(); 
    __syncthreads();    
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < 2; ++n) {
        gemm(lA(_, m, _), lB(_, n, _), rC(_, _, m, n));
      }
    }
    __syncthreads();
  }

  axpby(alpha, rC, beta, lC);
}
