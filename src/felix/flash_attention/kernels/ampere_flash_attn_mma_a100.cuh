#pragma once
#include "sm80_trait.cuh"
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/swizzle.hpp>
#include <math_constants.h>

namespace softmax {

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};

template <typename Op>
__device__ __forceinline__ float allreduce_4(float x, Op op) {
  constexpr unsigned mask = 0xffffffffu;

  x = op(x, __shfl_xor_sync(mask, x, 2));
  x = op(x, __shfl_xor_sync(mask, x, 1));

  return x;
}

template <bool ZeroInit, int MMA_M, int MMA_N, typename Op>
__device__ __forceinline__ void
thread_reduce_scores_rowwise(const float (&tSrS)[MMA_M][MMA_N][2][2][2],
                             float (&summary)[MMA_M][2], Op op) {
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int cm = 0; cm < 2; ++cm) {
      float v;

      if constexpr (ZeroInit) {
        v = tSrS[m][0][0][cm][0];
      } else {
        v = summary[m][cm];
      }

#pragma unroll
      for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
        for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            if constexpr (ZeroInit) {
              if (n != 0 || cn != 0 || c != 0) {
                v = op(v, tSrS[m][n][cn][cm][c]);
              }
            } else {
              v = op(v, tSrS[m][n][cn][cm][c]);
            }
          }
        }
      }

      summary[m][cm] = v;
    }
  }
}

template <bool ZeroInit, int MMA_M, int MMA_N>
__device__ __forceinline__ void
reduce_max_4lanes(const float (&tSrS)[MMA_M][MMA_N][2][2][2],
                  float (&row_max)[MMA_M][2]) {
  MaxOp op;

  thread_reduce_scores_rowwise<ZeroInit>(tSrS, row_max, op);

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int cm = 0; cm < 2; ++cm) {
      row_max[m][cm] = allreduce_4(row_max[m][cm], op);
    }
  }
}

template <bool ZeroInit, int MMA_M, int MMA_N>
__device__ __forceinline__ void
reduce_sum_local(const float (&tSrS)[MMA_M][MMA_N][2][2][2],
                 float (&row_sum)[MMA_M][2]) {
  SumOp op;

  thread_reduce_scores_rowwise<ZeroInit>(tSrS, row_sum, op);
}

template <int MMA_M>
__device__ __forceinline__ void
reduce_sum_4lanes_inplace(float (&row_sum)[MMA_M][2]) {
  SumOp op;

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int cm = 0; cm < 2; ++cm) {
      row_sum[m][cm] = allreduce_4(row_sum[m][cm], op);
    }
  }
}

template <int MMA_M, int MMA_N>
__device__ __forceinline__ void
scale_apply_exp2(float (&tSrS)[MMA_M][MMA_N][2][2][2],
                 const float (&row_max)[MMA_M][2], float softmax_scale_log2) {
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          const float max_scaled = row_max[m][cm] == -CUDART_INF_F
                                       ? 0.0f
                                       : row_max[m][cm] * softmax_scale_log2;

#pragma unroll
          for (int c = 0; c < 2; ++c) {
#ifdef UNFUSE_FMA
            tSrS[m][n][cn][cm][c] =
                exp2f(__fmul_rn(tSrS[m][n][cn][cm][c], softmax_scale_log2) -
                      max_scaled);
#else
            tSrS[m][n][cn][cm][c] =
                exp2f(tSrS[m][n][cn][cm][c] * softmax_scale_log2 - max_scaled);
#endif
          }
        }
      }
    }
  }
}

template <int MMA_M, int MMA_N> struct Softmax {
  float row_max[MMA_M][2];
  float row_sum[MMA_M][2];

  __device__ __forceinline__ Softmax() {}

  template <bool IsFirst, bool CheckInf = false, int MMA_K>
  __device__ __forceinline__ void
  softmax_rescale_o(float (&tSrS)[MMA_M][MMA_N][2][2][2],
                    float (&acc_o)[MMA_M][MMA_K][2][2][2],
                    float softmax_scale_log2) {
    if constexpr (IsFirst) {
      reduce_max_4lanes<true>(tSrS, row_max);
      scale_apply_exp2(tSrS, row_max, softmax_scale_log2);
      reduce_sum_local<true>(tSrS, row_sum);
    } else {
      float row_max_prev[MMA_M][2];

#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_max_prev[m][cm] = row_max[m][cm];
        }
      }

      reduce_max_4lanes<false>(tSrS, row_max);

#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          float row_max_cur;

          if constexpr (CheckInf) {
            row_max_cur =
                row_max[m][cm] == -CUDART_INF_F ? 0.0f : row_max[m][cm];
          } else {
            row_max_cur = row_max[m][cm];
          }

          const float scores_scale =
              exp2f((row_max_prev[m][cm] - row_max_cur) * softmax_scale_log2);
          row_sum[m][cm] *= scores_scale;

#pragma unroll
          for (int k = 0; k < MMA_K; ++k) {
#pragma unroll
            for (int ck = 0; ck < 2; ++ck) {
#pragma unroll
              for (int c = 0; c < 2; ++c) {
                acc_o[m][k][ck][cm][c] *= scores_scale;
              }
            }
          }
        }
      }

      scale_apply_exp2(tSrS, row_max, softmax_scale_log2);
      reduce_sum_local<false>(tSrS, row_sum);
    }
  }

  template <int MMA_K>
  __device__ __forceinline__ void
  normalize(float (&acc_o)[MMA_M][MMA_K][2][2][2]) {
    reduce_sum_4lanes_inplace(row_sum);

#pragma unroll
    for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
      for (int cm = 0; cm < 2; ++cm) {
        const float sum = row_sum[m][cm];
        const float inv_sum = (sum == 0.0f || sum != sum) ? 1.0f : 1.0f / sum;

#pragma unroll
        for (int k = 0; k < MMA_K; ++k) {
#pragma unroll
          for (int ck = 0; ck < 2; ++ck) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
              acc_o[m][k][ck][cm][c] *= inv_sum;
            }
          }
        }
      }
    }
  }
};

} // namespace softmax
template <typename T, std::size_t... Dims> struct tensor_type;

template <typename T> struct tensor_type<T> {
  using type = T;
};

template <typename T, std::size_t N, std::size_t... Rest>
struct tensor_type<T, N, Rest...> {
  using type = typename tensor_type<T, Rest...>::type[N];
};

template <typename T, std::size_t... Dims>
using tensor_t = typename tensor_type<T, Dims...>::type;

template <std::size_t... Dims, typename T>
__host__ __device__ constexpr auto as_tensor(T *ptr) {
  return reinterpret_cast<tensor_t<T, Dims...> *>(ptr);
}

template <std::size_t... Dims, typename T>
__host__ __device__ constexpr auto as_tensor(const T *ptr) {
  return reinterpret_cast<const tensor_t<T, Dims...> *>(ptr);
}

template <int kHeadDim> struct FlashFwdParams {
  const half *q; // [B, Sq, Hq, D]
  const half *k; // [B, Sk, Hk, D]
  const half *v; // [B, Sk, Hk, D]
  half *o;       // [B, Sq, Hq, D]

  int batch_size;

  int seqlen_q;
  int seqlen_k;

  int heads_q;
  int heads_k;
  int q_heads_per_kv_head; // heads_q / heads_k

  int q_batch_stride;
  int q_row_stride;
  int q_head_stride;

  int k_batch_stride;
  int k_row_stride;
  int k_head_stride;

  int v_batch_stride;
  int v_row_stride;
  int v_head_stride;

  int o_batch_stride;
  int o_row_stride;
  int o_head_stride;

  float softmax_scale_log2;
};

// using TiledMma = TiledMMA<
//     typename Base::MMA_Atom_Arch,
//     Layout<Shape<Int<kNWarps>, _1, _1>>,
//     Tile<Int<16 * kNWarps>, _16, _16>
// >;
template <int kWarps> struct TiledMMA {
  static constexpr int M = 16 * kWarps;
  static constexpr int N = 16;
  static constexpr int K = 16;
};

// Current block:
//
//   Q_i : [BM, D]
//   K_j : [BN, D]
//   V_j : [BN, D]
//
// Compute score:
//
//   acc_s = Q_i * K_j^T
//   acc_s *= softmax_scale_log2
//
// Online softmax:
//
//   m_new = max(m, rowmax(acc_s))
//
//   acc_s = exp2(acc_s - m_new)
//
// Now acc_s represents unnormalized P_j:
//
//   P_j = exp2(S2_j - m_new)
//
// Rescale old output:
//
//   alpha = exp2(m - m_new)
//
//   l     = alpha * l + rowsum(P_j)
//   acc_o = alpha * acc_o + P_j * V_j
//   m     = m_new
//
// After the last K/V block:
//
//   acc_o = acc_o / l
//
// Store acc_o to O.

template <int kHeadDim, int kBlockX, int kThreads>
__device__ __forceinline__ void async_load_x_tensor(half *sX, const half *gX,
                                                    int x_row_stride) {
  constexpr int kElementsPerAccess = 8;
  static_assert(kBlockX > 0);
  static_assert(kHeadDim % kElementsPerAccess == 0);

  constexpr int kVecsPerRow = kHeadDim / kElementsPerAccess;
  static_assert(kThreads % kVecsPerRow == 0);

  constexpr int kRowsPerIter = kThreads / kVecsPerRow;
  static_assert(kRowsPerIter > 0);
  static_assert(kBlockX % kRowsPerIter == 0);

  // Preconditions: sX/gX are 16B-aligned, and x_row_stride is a multiple of
  // 8 half elements. cp.async.cg<16> has no guarded slow path here.
  const int tXcol = threadIdx.x % kVecsPerRow;
  const int tXrow = threadIdx.x / kVecsPerRow;

#pragma unroll
  for (int i = 0; i < kBlockX / kRowsPerIter; ++i) {
    const int row = i * kRowsPerIter + tXrow;
    const int col = tXcol * kElementsPerAccess;

    cp_async::cg<16>(&sX[row * kHeadDim + col], &gX[row * x_row_stride + col]);
  }

  cp_async::commit_group();
}

template <int kHeadDim, int kBlockM, int kBlockN, int MMA_M, int MMA_N,
          int MMA_K>
__device__ __forceinline__ void compute_score_ss(half *sQ, half *sK,
                                                 half *tSrQ_p, half *tSrK_p,
                                                 float *tSrS_p) {
  static_assert(kHeadDim % 16 == 0);
  static_assert(MMA_K == kHeadDim / 16);
  auto tSrQ = as_tensor<MMA_K, 2, 2, 2>(tSrQ_p);
  auto tSrK = as_tensor<MMA_K, 2, 2, 2>(tSrK_p);
  auto tSrS = as_tensor<MMA_N, 2, 2, 2>(tSrS_p);
  int lane_id = threadIdx.x % 32;
  int tSsQ_row = lane_id % 16;
  int tSsQ_col = lane_id / 16;
  int tSsK_row = (lane_id / 16) * 8 + lane_id % 8;
  int tSsK_col = (lane_id % 16) / 8;

  int warp_id = threadIdx.x / 32;
  constexpr int kQSmemStride = kHeadDim;
  constexpr int kKSmemStride = kHeadDim;
#pragma unroll
  for (int m = 0; m < MMA_M; m++) {
    constexpr int k = 0;
    ldsm::x4<ldsm::N>(
        tSrQ[m][k][0][0], tSrQ[m][k][0][1], tSrQ[m][k][1][0], tSrQ[m][k][1][1],
        &sQ[(m * (kBlockM / MMA_M) + warp_id * 16 + tSsQ_row) * kQSmemStride +
            tSsQ_col * 8 + k * 16]);
  }
#pragma unroll
  for (int n = 0; n < MMA_N; n++) {
    constexpr int k = 0;
    ldsm::x4<ldsm::N>(tSrK[n][k][0][0], tSrK[n][k][0][1], tSrK[n][k][1][0],
                      tSrK[n][k][1][1],
                      &sK[(n * (kBlockN / MMA_N) + tSsK_row) * kKSmemStride +
                          tSsK_col * 8 + k * 16]);
  }
#pragma unroll
  for (int k = 0; k < MMA_K; k++) {
    if (k < MMA_K - 1) {
      int k_next = k + 1;
#pragma unroll
      for (int m = 0; m < MMA_M; m++) {
        ldsm::x4<ldsm::N>(
            tSrQ[m][k_next][0][0], tSrQ[m][k_next][0][1], tSrQ[m][k_next][1][0],
            tSrQ[m][k_next][1][1],
            &sQ[(m * (kBlockM / MMA_M) + warp_id * 16 + tSsQ_row) *
                    kQSmemStride +
                tSsQ_col * 8 + k_next * 16]);
      }
#pragma unroll
      for (int n = 0; n < MMA_N; n++) {
        ldsm::x4<ldsm::N>(
            tSrK[n][k_next][0][0], tSrK[n][k_next][0][1], tSrK[n][k_next][1][0],
            tSrK[n][k_next][1][1],
            &sK[(n * (kBlockN / MMA_N) + tSsK_row) * kKSmemStride +
                tSsK_col * 8 + k_next * 16]);
      }
    }

#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
#pragma unroll
      for (int n = 0; n < MMA_N; n++) {
        mma::m16n8k16_f32f16f16f32_accum(
            tSrS[m][n][0][0][0], tSrS[m][n][0][0][1], tSrS[m][n][0][1][0],
            tSrS[m][n][0][1][1], tSrQ[m][k][0][0], tSrQ[m][k][0][1],
            tSrQ[m][k][1][0], tSrQ[m][k][1][1], tSrK[n][k][0][0],
            tSrK[n][k][0][1]);

        mma::m16n8k16_f32f16f16f32_accum(
            tSrS[m][n][1][0][0], tSrS[m][n][1][0][1], tSrS[m][n][1][1][0],
            tSrS[m][n][1][1][1], tSrQ[m][k][0][0], tSrQ[m][k][0][1],
            tSrQ[m][k][1][0], tSrQ[m][k][1][1], tSrK[n][k][1][0],
            tSrK[n][k][1][1]);
      }
    }
  }
}

template <int kHeadDim, int kBlockM, int kBlockN, int MMA_M, int MMA_N,
          int MMA_K>
__device__ __forceinline__ void compute_output_rs(half *tOrP_p, half *sV,
                                                  half *tOrV_p, float *tOrO_p) {
  static_assert(kBlockN % 16 == 0);
  static_assert(MMA_N == kBlockN / 16);
  auto tOrP = as_tensor<MMA_N, 2, 2, 2>(tOrP_p);
  auto tOrO = as_tensor<MMA_K, 2, 2, 2>(tOrO_p);
  auto tOrV = as_tensor<MMA_N, 2, 2, 2>(tOrV_p);
  int lane_id = threadIdx.x % 32;
  int tOsV_row = lane_id % 16;
  int tOsV_col = (lane_id / 16) * 8;

  constexpr int kVSmemStride = kHeadDim;

#pragma unroll
  for (int k = 0; k < MMA_K; k++) {
    constexpr int n = 0;
    ldsm::x4<ldsm::T>(tOrV[k][n][0][0], tOrV[k][n][0][1], tOrV[k][n][1][0],
                      tOrV[k][n][1][1],
                      &sV[k * (kHeadDim / MMA_K) + tOsV_col +
                          (tOsV_row + n * 16) * kVSmemStride]);
  }

#pragma unroll
  for (int n = 0; n < MMA_N; n++) {
    if (n < MMA_N - 1) {
      int n_next = n + 1;
#pragma unroll
      for (int k = 0; k < MMA_K; k++) {
        ldsm::x4<ldsm::T>(tOrV[k][n_next][0][0], tOrV[k][n_next][0][1],
                          tOrV[k][n_next][1][0], tOrV[k][n_next][1][1],
                          &sV[k * (kHeadDim / MMA_K) + tOsV_col +
                              (tOsV_row + n_next * 16) * kVSmemStride]);
      }
    }
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
#pragma unroll
      for (int k = 0; k < MMA_K; k++) {
        mma::m16n8k16_f32f16f16f32_accum(
            tOrO[m][k][0][0][0], tOrO[m][k][0][0][1], tOrO[m][k][0][1][0],
            tOrO[m][k][0][1][1], tOrP[m][n][0][0], tOrP[m][n][0][1],
            tOrP[m][n][1][0], tOrP[m][n][1][1], tOrV[k][n][0][0],
            tOrV[k][n][0][1]);
        mma::m16n8k16_f32f16f16f32_accum(
            tOrO[m][k][1][0][0], tOrO[m][k][1][0][1], tOrO[m][k][1][1][0],
            tOrO[m][k][1][1][1], tOrP[m][n][0][0], tOrP[m][n][0][1],
            tOrP[m][n][1][0], tOrP[m][n][1][1], tOrV[k][n][1][0],
            tOrV[k][n][1][1]);
      }
    }
  }
}

template <int MMA_M, int MMA_K>
__device__ __forceinline__ void clearO(float (&tOrO)[MMA_M][MMA_K][2][2][2]) {
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int k = 0; k < MMA_K; ++k) {
#pragma unroll
      for (int ck = 0; ck < 2; ++ck) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            tOrO[m][k][ck][cm][c] = 0.0f;
          }
        }
      }
    }
  }
}

template <int MMA_M, int MMA_N>
__device__ __forceinline__ void clearS(float (&tSrS)[MMA_M][MMA_N][2][2][2]) {
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            tSrS[m][n][cn][cm][c] = 0.0f;
          }
        }
      }
    }
  }
}

template <int MMA_M, int MMA_N>
__device__ __forceinline__ void
convertS(float const (&tSrS)[MMA_M][MMA_N][2][2][2],
         half (&tOrP)[MMA_M][MMA_N][2][2][2]) {
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          const __half2 p =
              __floats2half2_rn(tSrS[m][n][cn][cm][0], tSrS[m][n][cn][cm][1]);
          as_u32_ref(tOrP[m][n][cn][cm]) =
              *reinterpret_cast<const uint32_t *>(&p);
        }
      }
    }
  }
}

__device__ __forceinline__ uint32_t fav2_pack_f32x2_to_f16x2(float x, float y) {
  const __half2 xy = __floats2half2_rn(x, y);
  return *reinterpret_cast<const uint32_t *>(&xy);
}

template <int kHeadDim, int kBlockM, int kWarps, int kThreads, int MMA_M,
          int MMA_K>
__device__ __forceinline__ void
store_output_epilogue(half *sO, half *gO, int o_row_stride,
                      const float (&tOrO)[MMA_M][MMA_K][2][2][2]) {
  constexpr int kElementsPerStore = 8;
  constexpr int kSmemStrideO = kHeadDim;
  constexpr int kStoreVecs = kBlockM * kHeadDim / kElementsPerStore;
  constexpr int kStoreIters = kStoreVecs / kThreads;
  static_assert(kHeadDim % kElementsPerStore == 0);
  static_assert(kThreads > 0);
  static_assert(kStoreVecs % kThreads == 0);
  static_assert(kStoreIters > 0);
  static_assert(kBlockM % (16 * kWarps) == 0);
  static_assert(MMA_M == kBlockM / (16 * kWarps));
  static_assert(MMA_K == kHeadDim / 16);

  __syncthreads();

  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int core_matrix_row = lane_id / 4;
  const int core_matrix_col = lane_id % 4;

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int k = 0; k < MMA_K; ++k) {
#pragma unroll
      for (int ck = 0; ck < 2; ++ck) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          const int row =
              m * (kBlockM / MMA_M) + warp_id * 16 + cm * 8 + core_matrix_row;
          const int col = k * 16 + ck * 8 + core_matrix_col * 2;

          *reinterpret_cast<uint32_t *>(&sO[row * kSmemStrideO + col]) =
              fav2_pack_f32x2_to_f16x2(tOrO[m][k][ck][cm][0],
                                       tOrO[m][k][ck][cm][1]);
        }
      }
    }
  }

  __syncthreads();

  // Preconditions: gO is 16B-aligned, o_row_stride is a multiple of 8 half
  // elements, and this kernel is launched only for full Q-row blocks.
#pragma unroll
  for (int i = 0; i < kStoreIters; ++i) {
    const int vec = i * kThreads + threadIdx.x;
    const int row = vec / (kHeadDim / kElementsPerStore);
    const int col = (vec % (kHeadDim / kElementsPerStore)) * kElementsPerStore;
    const uint4 *s_ptr =
        reinterpret_cast<const uint4 *>(&sO[row * kSmemStrideO + col]);
    uint4 *g_ptr = reinterpret_cast<uint4 *>(&gO[row * o_row_stride + col]);

    *g_ptr = *s_ptr;
  }
}

template <int kHeadDim, int kBlockM, int kBlockN>
__device__ __forceinline__ void
compute_attn_1rowblock(const FlashFwdParams<kHeadDim> &params, const int bidb,
                       const int bidh, const int m_block) {
  constexpr int kWarps = 4;
  constexpr int kThreads = kWarps * 32;
  static_assert(kHeadDim % 16 == 0);
  static_assert(kBlockM > 0);
  static_assert(kBlockN > 0);
  static_assert(kBlockM % (16 * kWarps) == 0);
  static_assert(kBlockN % 16 == 0);

  const int QueryBatchStride = params.q_batch_stride;
  const int QueryRowStride = params.q_row_stride;
  const int QueryHeadStride = params.q_head_stride;

  const int KeyBatchStride = params.k_batch_stride;
  const int KeyRowStride = params.k_row_stride;
  const int KeyHeadStride = params.k_head_stride;

  const int ValueBatchStride = params.v_batch_stride;
  const int ValueRowStride = params.v_row_stride;
  const int ValueHeadStride = params.v_head_stride;

  const int OutputBatchStride = params.o_batch_stride;
  const int OutputRowStride = params.o_row_stride;
  const int OutputHeadStride = params.o_head_stride;

  constexpr int MMA_M = kBlockM / (16 * kWarps);
  constexpr int MMA_N = kBlockN / 16;
  constexpr int MMA_K = kHeadDim / 16;

  // Launch/params preconditions:
  // - blockDim.x == kThreads.
  // - grid = (params.seqlen_q / kBlockM, params.batch_size, params.heads_q).
  // - params.seqlen_q % kBlockM == 0 and params.seqlen_k % kBlockN == 0.
  // - params.heads_q == params.heads_k * params.q_heads_per_kv_head.
  // - Q/K/V/O base pointers are non-null and 16B-aligned.
  // - Q/K/V/O row strides are positive and multiples of 8 half elements.

  // (MMA_M, MMA_K, CoreK, CoreM, Core)
  half tSrQ[MMA_M][MMA_K][2][2][2];
  // (MMA_N, MMA_K, CoreN, CoreK, Core)
  half tSrK[MMA_N][MMA_K][2][2][2];
  // (MMA_M, MMA_N, CoreN, CoreM, Core)
  float tSrS[MMA_M][MMA_N][2][2][2];

  // (MMA_M, MMA_N, CoreN, CoreM, Core)
  half tOrP[MMA_M][MMA_N][2][2][2];
  // (MMA_K, MMA_N, CoreK, CoreN, Core)
  half tOrV[MMA_K][MMA_N][2][2][2];
  // (MMA_M, MMA_K, CoreK, CoreM, Core)
  float tOrO[MMA_M][MMA_K][2][2][2];

  using TiledMma = TiledMMA<kWarps>;
  const int kv_head = bidh / params.q_heads_per_kv_head;

  // Tensor gQ (bidb, m_block, bidh/kv_head, _)
  // Tensor gK, gV (bidb, _, bidh/kv_head, _)
  auto gQ = params.q + bidb * QueryBatchStride + bidh * QueryHeadStride +
            m_block * kBlockM * QueryRowStride;
  auto gK_base = params.k + bidb * KeyBatchStride + kv_head * KeyHeadStride;
  auto gV_base = params.v + bidb * ValueBatchStride + kv_head * ValueHeadStride;
  auto gO = params.o + bidb * OutputBatchStride + bidh * OutputHeadStride +
            m_block * kBlockM * OutputRowStride;

  extern __shared__ half smem[];

  half *sQ = smem;
  half *sK = sQ + kBlockM * kHeadDim;
  half *sV = sK + kBlockN * kHeadDim;

  async_load_x_tensor<kHeadDim, kBlockM, kThreads>(sQ, gQ, QueryRowStride);
  async_load_x_tensor<kHeadDim, kBlockN, kThreads>(sK, gK_base, KeyRowStride);

  clearO(tOrO);

  // Wait Q and K_0 ready.
  cp_async::wait_group<0>();
  __syncthreads();
  const int num_k_tiles = params.seqlen_k / kBlockN;
  softmax::Softmax<MMA_M, MMA_N> softmax;

  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    const half *gV = gV_base + k_tile * kBlockN * ValueRowStride;


    // Load V_current.
    async_load_x_tensor<kHeadDim, kBlockN, kThreads>(sV, gV, ValueRowStride);


    // acc_s = Q_i @ K_current^T
    clearS(tSrS);

    compute_score_ss<kHeadDim, kBlockM, kBlockN, MMA_M, MMA_N, MMA_K>(
        sQ, sK, &tSrQ[0][0][0][0][0], &tSrK[0][0][0][0][0],
        &tSrS[0][0][0][0][0]);


    // Need V_current before P @ V.
    cp_async::wait_group<0>();
    __syncthreads();

    // Current K is no longer needed after compute_score_ss().
    if (k_tile + 1 < num_k_tiles) {
      const half *gK_next = gK_base + (k_tile + 1) * kBlockN * KeyRowStride;

      async_load_x_tensor<kHeadDim, kBlockN, kThreads>(sK, gK_next,
                                                       KeyRowStride);
    }

    // softmax online + acc_o += P @ V_current
    if (k_tile == 0) {
      softmax.softmax_rescale_o<true>(tSrS, tOrO, params.softmax_scale_log2);
    } else {
      softmax.softmax_rescale_o<false>(tSrS, tOrO, params.softmax_scale_log2);
    }
    convertS(tSrS, tOrP);
    compute_output_rs<kHeadDim, kBlockM, kBlockN, MMA_M, MMA_N, MMA_K>(
        &tOrP[0][0][0][0][0], sV, &tOrV[0][0][0][0][0],
        &tOrO[0][0][0][0][0]);

    // Before next iteration, K_next must be ready in sK.
    if (k_tile + 1 < num_k_tiles) {
      cp_async::wait_group<0>();
      __syncthreads();
    }
  }

  softmax.normalize(tOrO);
  store_output_epilogue<kHeadDim, kBlockM, kWarps, kThreads, MMA_M, MMA_K>(
      sQ, gO, OutputRowStride, tOrO);
}

template <int kHeadDim, int kBlockM, int kBlockN>
__global__ void flash_attn_v2(FlashFwdParams<kHeadDim> params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;

  compute_attn_1rowblock<kHeadDim, kBlockM, kBlockN>(params, bidb, bidh,
                                                     m_block);
}

namespace fav2_sm80 {

template <int kHeadDim, int kBlockM, int kBlockN>
struct FlashAttnV2LaunchConfig {
  static constexpr int kHeadDimValue = kHeadDim;
  static constexpr int kBlockMValue = kBlockM;
  static constexpr int kBlockNValue = kBlockN;
  static constexpr int kWarps = 4;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kSmemBytes =
      (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(half);
};

template <int kHeadDim> struct FlashAttnV2A100Config {
  static constexpr bool kSupported = false;
  static constexpr int kBlockMValue = 0;
  static constexpr int kBlockNValue = 0;
};

// Wired subset of the official FlashAttention SM80/A100 non-dropout configs.
// The official hdim96 config also uses 128x64x4, but this loader currently
// requires kThreads % (kHeadDim / 8) == 0. Official hdim192/256 use 8 warps on
// A100, while this handwritten kernel is fixed at 4 warps.
template <> struct FlashAttnV2A100Config<32>
    : FlashAttnV2LaunchConfig<32, 128, 128> {
  static constexpr bool kSupported = true;
};

template <> struct FlashAttnV2A100Config<64>
    : FlashAttnV2LaunchConfig<64, 128, 128> {
  static constexpr bool kSupported = true;
};

template <> struct FlashAttnV2A100Config<128>
    : FlashAttnV2LaunchConfig<128, 128, 64> {
  static constexpr bool kSupported = true;
};

template <int kHeadDim, int kBlockM, int kBlockN>
inline cudaError_t
launch_flash_attn_v2_config(FlashFwdParams<kHeadDim> params,
                            cudaStream_t stream = 0) {
  using Config = FlashAttnV2LaunchConfig<kHeadDim, kBlockM, kBlockN>;
  static_assert(kHeadDim >= 16);
  static_assert(kHeadDim % 16 == 0);

  constexpr int kElementsPerAccess = 8;
  static_assert(kHeadDim % kElementsPerAccess == 0);

  constexpr int kVecsPerRow = kHeadDim / kElementsPerAccess;
  constexpr int kRowsPerLoadIter = Config::kThreads / kVecsPerRow;

  static_assert(Config::kThreads % kVecsPerRow == 0);
  static_assert(kBlockM % kRowsPerLoadIter == 0);
  static_assert(kBlockN % kRowsPerLoadIter == 0);
  static_assert((kBlockM * kHeadDim / kElementsPerAccess) %
                    Config::kThreads ==
                0);
  static_assert(kBlockM % (16 * Config::kWarps) == 0);
  static_assert(kBlockN % 16 == 0);

  if (params.seqlen_q % kBlockM != 0 || params.seqlen_k % kBlockN != 0 ||
      params.batch_size <= 0 || params.heads_q <= 0 || params.heads_k <= 0 ||
      params.q_heads_per_kv_head <= 0 ||
      params.heads_k * params.q_heads_per_kv_head != params.heads_q) {
    return cudaErrorInvalidValue;
  }

  auto kernel_fptr = flash_attn_v2<kHeadDim, kBlockM, kBlockN>;

  if constexpr (Config::kSmemBytes >= 48 * 1024) {
    cudaError_t err = cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
        Config::kSmemBytes);
    if (err != cudaSuccess) {
      return err;
    }
  }

  dim3 block(Config::kThreads);
  dim3 grid(params.seqlen_q / kBlockM, params.batch_size, params.heads_q);

  kernel_fptr<<<grid, block, Config::kSmemBytes, stream>>>(params);
  return cudaGetLastError();
}

template <int kHeadDim>
inline cudaError_t launch_flash_attn_v2_a100(FlashFwdParams<kHeadDim> params,
                                             cudaStream_t stream = 0) {
  using Config = FlashAttnV2A100Config<kHeadDim>;
  static_assert(
      Config::kSupported,
      "A100 launch config is wired for head_dim 32, 64, and 128 in this "
      "handwritten kernel.");

  return launch_flash_attn_v2_config<kHeadDim, Config::kBlockMValue,
                                     Config::kBlockNValue>(params, stream);
}

} // namespace fav2_sm80
