#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_impl.hpp"
#include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cute/tensor.hpp>

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t WARP_NUMS = 2;
constexpr uint32_t THREAD_NUMS = WARP_NUMS * WARP_SIZE;
template <class ProblemShape, class CtaTiler, class AStride, class ASmemLayout,
          class AThreadLayout, class BStride, class BSmemLayout,
          class BThreadLayout, class CStride, class CSmemLayout,
          class CThreadLayout>
__global__ static __launch_bounds__(
    THREAD_NUMS) void cute_ampere_sgemm_64x64_nn_k(ProblemShape shape_MNK,
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
                                                   CThreadLayout tC,
                                                   float alpha, float beta) {
  using namespace cute;
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord);
}