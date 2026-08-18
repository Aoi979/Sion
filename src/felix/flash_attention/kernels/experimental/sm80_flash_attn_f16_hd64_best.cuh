#pragma once
#include "../../detail/sm80_flash_attn_f16_hd64_best_trait.cuh"
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Experimental source snapshot specialized for the D64 benchmark target.
namespace felix::detail::sm80_flash_attn_f16_hd64_best {

#ifndef FAV2_D64_ISSUE_SCHEDULE
#define FAV2_D64_ISSUE_SCHEDULE 1
#endif

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

  // D64 issue-schedule variant: the final QK HMMA updates row_max as each
  // score becomes ready.  Keep the previous tile's max separate so the
  // online-softmax rescale does not have to recompute the current max.
  template <bool IsFirst, int MMA_K>
  __device__ __forceinline__ void softmax_prepare_o_from_precomputed_max(
      float (&acc_o)[MMA_M][MMA_K][2][2][2],
      const float (&row_max_prev)[MMA_M][2], float softmax_scale_log2) {
    MaxOp max_op;

    if constexpr (IsFirst) {
#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_max[m][cm] = allreduce_4(row_max[m][cm], max_op);
        }
      }
    } else {
#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_max[m][cm] = allreduce_4(row_max[m][cm], max_op);
          const float scores_scale =
              exp2f((row_max_prev[m][cm] - row_max[m][cm]) *
                    softmax_scale_log2);
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
    }
  }

  template <bool IsFirst, int MMA_K>
  __device__ __forceinline__ void softmax_rescale_o_from_precomputed_max(
      float (&tSrS)[MMA_M][MMA_N][2][2][2],
      float (&acc_o)[MMA_M][MMA_K][2][2][2],
      const float (&row_max_prev)[MMA_M][2], float softmax_scale_log2) {
    MaxOp max_op;

    if constexpr (IsFirst) {
#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_max[m][cm] = allreduce_4(row_max[m][cm], max_op);
        }
      }
      scale_apply_exp2(tSrS, row_max, softmax_scale_log2);
      reduce_sum_local<true>(tSrS, row_sum);
    } else {
#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_max[m][cm] = allreduce_4(row_max[m][cm], max_op);
          const float scores_scale =
              exp2f((row_max_prev[m][cm] - row_max[m][cm]) *
                    softmax_scale_log2);
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
        const float inv_sum = 1.0f / sum;

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

// Use a plain row-major shared-memory layout with eight half-element padding
// elements per row.  The +8 keeps each row 16-byte aligned while avoiding the
// address transform and XOR/swizzle instructions in the load path.
static constexpr int kFav2SmemPad = 8;

template <int kSmemStride>
__device__ __forceinline__ int fav2_padded_smem_offset(int row, int col) {
  return row * kSmemStride + col;
}

template <int kSmemStride>
__device__ __forceinline__ half *
fav2_padded_smem_ptr(half *ptr, int row, int col) {
  return ptr + fav2_padded_smem_offset<kSmemStride>(row, col);
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

// Runtime strides are useful for the general PyTorch entry point, but they
// also keep address arithmetic live in the hot path.  Keep a separate
// compile-time stride policy for fixed-shape experiments.  The params object
// is still passed to the kernel so this path can be wired into a benchmark
// later without duplicating the kernel body; all stride fields are ignored by
// the static policy.
struct Fav2RuntimeStrides {
  template <typename Params>
  __device__ __forceinline__ static int q_batch(const Params &p) {
    return p.q_batch_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int q_row(const Params &p) {
    return p.q_row_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int q_head(const Params &p) {
    return p.q_head_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int k_batch(const Params &p) {
    return p.k_batch_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int k_row(const Params &p) {
    return p.k_row_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int k_head(const Params &p) {
    return p.k_head_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int v_batch(const Params &p) {
    return p.v_batch_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int v_row(const Params &p) {
    return p.v_row_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int v_head(const Params &p) {
    return p.v_head_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int o_batch(const Params &p) {
    return p.o_batch_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int o_row(const Params &p) {
    return p.o_row_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int o_head(const Params &p) {
    return p.o_head_stride;
  }
  template <typename Params>
  __device__ __forceinline__ static int seqlen_k(const Params &p) {
    return p.seqlen_k;
  }
  template <typename Params>
  __device__ __forceinline__ static int q_heads_per_kv_head(
      const Params &p) {
    return p.q_heads_per_kv_head;
  }
  template <typename Params>
  __device__ __forceinline__ static float softmax_scale_log2(
      const Params &p) {
    return p.softmax_scale_log2;
  }
};

template <int QBatchStride, int QRowStride, int QHeadStride,
          int KBatchStride, int KRowStride, int KHeadStride,
          int VBatchStride, int VRowStride, int VHeadStride,
          int OBatchStride, int ORowStride, int OHeadStride>
struct Fav2StaticStrides {
  template <typename Params>
  __device__ __forceinline__ static int q_batch(const Params &) {
    return QBatchStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int q_row(const Params &) {
    return QRowStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int q_head(const Params &) {
    return QHeadStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int k_batch(const Params &) {
    return KBatchStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int k_row(const Params &) {
    return KRowStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int k_head(const Params &) {
    return KHeadStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int v_batch(const Params &) {
    return VBatchStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int v_row(const Params &) {
    return VRowStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int v_head(const Params &) {
    return VHeadStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int o_batch(const Params &) {
    return OBatchStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int o_row(const Params &) {
    return ORowStride;
  }
  template <typename Params>
  __device__ __forceinline__ static int o_head(const Params &) {
    return OHeadStride;
  }
};

// Fixed-shape config. All non-pointer launch metadata is an NTTP, including
// the tensor strides in half elements.
template <int HeadDim, int BlockM, int BlockN, int BatchSize, int SeqlenQ,
          int SeqlenK, int HeadsQ, int HeadsK, int QHeadsPerKVHead,
          int QBatchStride, int QRowStride,
          int QHeadStride, int KBatchStride, int KRowStride,
          int KHeadStride, int VBatchStride, int VRowStride,
          int VHeadStride, int OBatchStride, int ORowStride,
          int OHeadStride>
struct Fav2StaticConfig
    : Fav2StaticStrides<QBatchStride, QRowStride, QHeadStride, KBatchStride,
                        KRowStride, KHeadStride, VBatchStride, VRowStride,
                        VHeadStride, OBatchStride, ORowStride, OHeadStride> {
  static constexpr int kHeadDim = HeadDim;
  static constexpr int kBlockM = BlockM;
  static constexpr int kBlockN = BlockN;
  static constexpr int kBatchSize = BatchSize;
  static constexpr int kSeqlenQ = SeqlenQ;
  static constexpr int kSeqlenK = SeqlenK;
  static constexpr int kHeadsQ = HeadsQ;
  static constexpr int kHeadsK = HeadsK;
  static constexpr int kQHeadsPerKVHead = QHeadsPerKVHead;
  static constexpr float kSoftmaxScaleLog2 =
      HeadDim == 32   ? 0.25503486164919736F
      : HeadDim == 64 ? 0.18033688011112042F
                      : 0.12751743082459868F;

  template <typename Params>
  __device__ __forceinline__ static int seqlen_k(const Params &) {
    return kSeqlenK;
  }
  template <typename Params>
  __device__ __forceinline__ static int q_heads_per_kv_head(const Params &) {
    return kQHeadsPerKVHead;
  }
  template <typename Params>
  __device__ __forceinline__ static float softmax_scale_log2(const Params &) {
    return kSoftmaxScaleLog2;
  }
};

// [B, S, H, D] contiguous target used for the current A100 investigation:
// B=2, Sq=Sk=4096, Hq=Hk=8, D=64. Strides are in half elements.
using Fav2StaticB2Sq4096Sk4096H8D64Config =
    Fav2StaticConfig<64, 128, 64, 2, 4096, 4096, 8, 8, 1, 2097152, 64,
                     262144, 2097152, 64, 262144, 2097152, 64, 262144,
                     2097152, 64, 262144>;

// [B, S, H, D] contiguous target:
// B=1, Sq=4096, Sk=4032, Hq=Hk=16, D=128. Strides are in half elements.
using Fav2StaticB1Sq4096Sk4032H16D128Config =
    Fav2StaticConfig<128, 128, 64, 1, 4096, 4032, 16, 16, 1, 8388608,
                     128, 524288, 8257536, 128, 516096, 8257536, 128,
                     516096, 8388608, 128, 524288>;

// All scalar metadata is supplied by the template config.  Only data
// addresses remain as runtime kernel arguments.
struct Fav2StaticKernelPtrs {
  const half *q;
  const half *k;
  const half *v;
  half *o;
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

template <int kHeadDim, int kSmemStride, int kBlockX, int kThreads>
__device__ __forceinline__ void async_load_x_tensor(half *sX, const half *gX,
                                                    int x_row_stride) {
  constexpr int kElementsPerAccess = 8;
  static_assert(kBlockX > 0);
  static_assert(kHeadDim % kElementsPerAccess == 0);
  static_assert(kSmemStride == kHeadDim + kFav2SmemPad);

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

    cp_async::cg<16>(fav2_padded_smem_ptr<kSmemStride>(sX, row, col),
                     &gX[row * x_row_stride + col]);
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
  constexpr int kQSmemStride = kHeadDim + kFav2SmemPad;
  constexpr int kKSmemStride = kHeadDim + kFav2SmemPad;
#pragma unroll
  for (int m = 0; m < MMA_M; m++) {
    constexpr int k = 0;
    ldsm::x4<ldsm::N>(
        tSrQ[m][k][0][0], tSrQ[m][k][0][1], tSrQ[m][k][1][0],
        tSrQ[m][k][1][1],
        fav2_padded_smem_ptr<kQSmemStride>(
            sQ, m * (kBlockM / MMA_M) + warp_id * 16 + tSsQ_row,
            tSsQ_col * 8 + k * 16));
  }
#pragma unroll
  for (int n = 0; n < MMA_N; n++) {
    constexpr int k = 0;
    ldsm::x4<ldsm::N>(
        tSrK[n][k][0][0], tSrK[n][k][0][1], tSrK[n][k][1][0],
        tSrK[n][k][1][1],
        fav2_padded_smem_ptr<kKSmemStride>(
            sK, n * (kBlockN / MMA_N) + tSsK_row,
            tSsK_col * 8 + k * 16));
  }
#pragma unroll
  for (int k = 0; k < MMA_K; k++) {
    if (k < MMA_K - 1) {
      int k_next = k + 1;
#pragma unroll
      for (int m = 0; m < MMA_M; m++) {
        ldsm::x4<ldsm::N>(
            tSrQ[m][k_next][0][0], tSrQ[m][k_next][0][1],
            tSrQ[m][k_next][1][0], tSrQ[m][k_next][1][1],
            fav2_padded_smem_ptr<kQSmemStride>(
                sQ, m * (kBlockM / MMA_M) + warp_id * 16 + tSsQ_row,
                tSsQ_col * 8 + k_next * 16));
      }
#pragma unroll
      for (int n = 0; n < MMA_N; n++) {
        ldsm::x4<ldsm::N>(
            tSrK[n][k_next][0][0], tSrK[n][k_next][0][1],
            tSrK[n][k_next][1][0], tSrK[n][k_next][1][1],
            fav2_padded_smem_ptr<kKSmemStride>(
                sK, n * (kBlockN / MMA_N) + tSsK_row,
                tSsK_col * 8 + k_next * 16));
      }
    }

#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
#pragma unroll
      for (int n = 0; n < MMA_N; n++) {
        mma::m16n8k16_f32f16f16f32_accum(
            tSrS[m][n][0][0][0], tSrS[m][n][0][0][1],
            tSrS[m][n][0][1][0], tSrS[m][n][0][1][1], tSrQ[m][k][0][0],
            tSrQ[m][k][0][1], tSrQ[m][k][1][0], tSrQ[m][k][1][1],
            tSrK[n][k][0][0], tSrK[n][k][0][1]);
        mma::m16n8k16_f32f16f16f32_accum(
            tSrS[m][n][1][0][0], tSrS[m][n][1][0][1],
            tSrS[m][n][1][1][0], tSrS[m][n][1][1][1], tSrQ[m][k][0][0],
            tSrQ[m][k][0][1], tSrQ[m][k][1][0], tSrQ[m][k][1][1],
            tSrK[n][k][1][0], tSrK[n][k][1][1]);
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

  constexpr int kVSmemStride = kHeadDim + kFav2SmemPad;

#pragma unroll
  for (int k = 0; k < MMA_K; k++) {
    constexpr int n = 0;
    ldsm::x4<ldsm::T>(
        tOrV[k][n][0][0], tOrV[k][n][0][1], tOrV[k][n][1][0],
        tOrV[k][n][1][1],
        fav2_padded_smem_ptr<kVSmemStride>(
            sV, tOsV_row + n * 16, k * (kHeadDim / MMA_K) + tOsV_col));
  }

#pragma unroll
  for (int n = 0; n < MMA_N; n++) {
    if (n < MMA_N - 1) {
      int n_next = n + 1;
#pragma unroll
      for (int k = 0; k < MMA_K; k++) {
        ldsm::x4<ldsm::T>(
            tOrV[k][n_next][0][0], tOrV[k][n_next][0][1],
            tOrV[k][n_next][1][0], tOrV[k][n_next][1][1],
            fav2_padded_smem_ptr<kVSmemStride>(
                sV, tOsV_row + n_next * 16,
                k * (kHeadDim / MMA_K) + tOsV_col));
      }
    }
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
#pragma unroll
      for (int k = 0; k < MMA_K; k++) {
        mma::m16n8k16_f32f16f16f32_accum(
            tOrO[m][k][0][0][0], tOrO[m][k][0][0][1],
            tOrO[m][k][0][1][0], tOrO[m][k][0][1][1],
            tOrP[m][n][0][0], tOrP[m][n][0][1], tOrP[m][n][1][0],
            tOrP[m][n][1][1], tOrV[k][n][0][0], tOrV[k][n][0][1]);
        mma::m16n8k16_f32f16f16f32_accum(
            tOrO[m][k][1][0][0], tOrO[m][k][1][0][1],
            tOrO[m][k][1][1][0], tOrO[m][k][1][1][1],
            tOrP[m][n][0][0], tOrP[m][n][0][1], tOrP[m][n][1][0],
            tOrP[m][n][1][1], tOrV[k][n][1][0], tOrV[k][n][1][1]);
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

__device__ __forceinline__ void
fav2_store_global_no_allocate(uint4 *addr, uint4 value) {
  const longlong2 value64 = *reinterpret_cast<const longlong2 *>(&value);
  asm volatile("{\n\t .reg .b128 B128_src; \n\t"
               "mov.b128 B128_src, {%1, %2}; \n"
               "st.relaxed.sys.global.L1::no_allocate.b128 [%0], B128_src;\n\t"
               "}"
               :
               : "l"(addr), "l"(value64.x), "l"(value64.y)
               : "memory");
}

template <int kHeadDim, int kBlockM, int kWarps, int kThreads, int MMA_M,
          int MMA_K>
__device__ __forceinline__ void
store_output_epilogue(half *sO, half *gO, int o_row_stride,
                      const float (&tOrO)[MMA_M][MMA_K][2][2][2]) {
  constexpr int kElementsPerStore = 8;
  constexpr int kSmemStrideO = kHeadDim + kFav2SmemPad;
  constexpr int kStoreVecs = kBlockM * kHeadDim / kElementsPerStore;
  constexpr int kStoreIters = kStoreVecs / kThreads;
  static_assert(kHeadDim % kElementsPerStore == 0);
  static_assert(kSmemStrideO % kElementsPerStore == 0);
  static_assert(kThreads > 0);
  static_assert(kStoreVecs % kThreads == 0);
  static_assert(kStoreIters > 0);
  static_assert(kBlockM % (16 * kWarps) == 0);
  static_assert(MMA_M == kBlockM / (16 * kWarps));
  static_assert(MMA_K == kHeadDim / 16);

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

    fav2_store_global_no_allocate(g_ptr, *s_ptr);
  }
}

template <int kHeadDim, int kBlockM, int kBlockN,
          typename Strides = Fav2RuntimeStrides,
          typename Params = FlashFwdParams<kHeadDim>>
__device__ __forceinline__ void
compute_attn_1rowblock(const Params &params, const int bidb, const int bidh,
                       const int m_block) {
  constexpr int kWarps = 4;
  constexpr int kThreads = kWarps * 32;
  static_assert(kHeadDim % 16 == 0);
  static_assert(kBlockM > 0);
  static_assert(kBlockN > 0);
  static_assert(kBlockM % (16 * kWarps) == 0);
  static_assert(kBlockN % 16 == 0);

  const int QueryBatchStride = Strides::q_batch(params);
  const int QueryRowStride = Strides::q_row(params);
  const int QueryHeadStride = Strides::q_head(params);

  const int KeyBatchStride = Strides::k_batch(params);
  const int KeyRowStride = Strides::k_row(params);
  const int KeyHeadStride = Strides::k_head(params);

  const int ValueBatchStride = Strides::v_batch(params);
  const int ValueRowStride = Strides::v_row(params);
  const int ValueHeadStride = Strides::v_head(params);

  const int OutputBatchStride = Strides::o_batch(params);
  const int OutputRowStride = Strides::o_row(params);
  const int OutputHeadStride = Strides::o_head(params);

  constexpr int MMA_M = kBlockM / (16 * kWarps);
  constexpr int MMA_N = kBlockN / 16;
  constexpr int MMA_K = kHeadDim / 16;
  constexpr int kQKVSmemStride = kHeadDim + kFav2SmemPad;

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

  clearO(tOrO);

  using TiledMma = TiledMMA<kWarps>;
  const int kv_head = bidh / Strides::q_heads_per_kv_head(params);

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
  half *sK = sQ + kBlockM * kQKVSmemStride;
  half *sV = sK + kBlockN * kQKVSmemStride;

  const int num_k_tiles = Strides::seqlen_k(params) / kBlockN;
  const int initial_k_tile = num_k_tiles - 1;

  async_load_x_tensor<kHeadDim, kQKVSmemStride, kBlockM, kThreads>(
      sQ, gQ, QueryRowStride);
  async_load_x_tensor<kHeadDim, kQKVSmemStride, kBlockN, kThreads>(
      sK, gK_base + initial_k_tile * kBlockN * KeyRowStride, KeyRowStride);

  // Wait Q and the initial (last) K tile ready.
  cp_async::wait_group<0>();
  __syncthreads();
  softmax::Softmax<MMA_M, MMA_N> softmax;

#if FAV2_D64_ISSUE_SCHEDULE
  // Q is invariant across the K-tile loop.  Load its register fragments once
  // after the initial Q/K barrier and keep them live for every score tile.
  // Reloading Q here inside the loop costs eight LDSM.M88 instructions per
  // K-tile even though only K changes from one iteration to the next.
  if constexpr (kHeadDim == 64 && kBlockN == 64) {
#define FAV2_D64_LOAD_Q(M, K) \
    ldsm::x4<ldsm::N>( \
        tSrQ[M][K][0][0], tSrQ[M][K][0][1], tSrQ[M][K][1][0], \
        tSrQ[M][K][1][1], \
        fav2_padded_smem_ptr<kQKVSmemStride>( \
            sQ, M * (kBlockM / MMA_M) + (threadIdx.x / 32) * 16 + \
                (threadIdx.x % 32) % 16, \
            ((threadIdx.x % 32) / 16) * 8 + K * 16))
    FAV2_D64_LOAD_Q(0, 0); FAV2_D64_LOAD_Q(1, 0);
    FAV2_D64_LOAD_Q(0, 1); FAV2_D64_LOAD_Q(1, 1);
    FAV2_D64_LOAD_Q(0, 2); FAV2_D64_LOAD_Q(1, 2);
    FAV2_D64_LOAD_Q(0, 3); FAV2_D64_LOAD_Q(1, 3);
#undef FAV2_D64_LOAD_Q
  }
#endif

  for (int k_tile = initial_k_tile; k_tile >= 0; --k_tile) {
    // K_prev was issued by the previous iteration. Wait at the point where it
    // becomes the current K tile, keeping the wait out of the previous
    // iteration's output path.
    if (k_tile != initial_k_tile) {
      cp_async::wait_group<0>();
      __syncthreads();
    }

    const half *gV = gV_base + k_tile * kBlockN * ValueRowStride;

    // --------------------------------------------------------
    // Load V_current.
    // This overlaps with QK^T.
    // --------------------------------------------------------
    async_load_x_tensor<kHeadDim, kQKVSmemStride, kBlockN, kThreads>(
        sV, gV, ValueRowStride);

    // --------------------------------------------------------
    // Current score:
    //   acc_s = Q_i @ K_current^T
    // Uses sQ and sK.
    // --------------------------------------------------------
    float d64_o_scale[2][2];
#if FAV2_D64_ISSUE_SCHEDULE
    if constexpr (kHeadDim == 64 && kBlockN == 64) {
      static_assert(MMA_M == 2 && MMA_N == 4 && MMA_K == 4);
      // Keep scale and max as separate issueable stages.  The fused log2
      // constant preserves the baseline's online-softmax state domain while
      // allowing the completed HMMA accumulator to be overwritten first.
      constexpr float d64_score_log2e = 0.1803368777036667F;

      float d64_prev_max[2][2];
      d64_prev_max[0][0] = softmax.row_max[0][0];
      d64_prev_max[0][1] = softmax.row_max[0][1];
      d64_prev_max[1][0] = softmax.row_max[1][0];
      d64_prev_max[1][1] = softmax.row_max[1][1];
      if (k_tile == initial_k_tile) {
        softmax.row_max[0][0] = -CUDART_INF_F;
        softmax.row_max[0][1] = -CUDART_INF_F;
        softmax.row_max[1][0] = -CUDART_INF_F;
        softmax.row_max[1][1] = -CUDART_INF_F;
      }

#define FAV2_D64_LOAD_K(NN, K) \
      ldsm::x4<ldsm::N>( \
          tSrK[NN][K][0][0], tSrK[NN][K][0][1], tSrK[NN][K][1][0], \
          tSrK[NN][K][1][1], \
          fav2_padded_smem_ptr<kQKVSmemStride>( \
              sK, NN * (kBlockN / MMA_N) + ((threadIdx.x % 32) / 16) * 8 + \
                  (threadIdx.x % 32) % 8, \
              (((threadIdx.x % 32) % 16) / 8) * 8 + K * 16))
#define FAV2_D64_MMA(M, N, CN, K) \
      mma::m16n8k16_f32f16f16f32_accum( \
          tSrS[M][N][CN][0][0], tSrS[M][N][CN][0][1], \
          tSrS[M][N][CN][1][0], tSrS[M][N][CN][1][1], \
          tSrQ[M][K][0][0], tSrQ[M][K][0][1], tSrQ[M][K][1][0], \
          tSrQ[M][K][1][1], tSrK[N][K][CN][0], tSrK[N][K][CN][1])
#define FAV2_D64_MMA_AFTER_MAX(M, N, CN, K, DM, CM) \
      do { \
        uint32_t d64_mma_b0; \
        asm volatile("lop3.b32 %0, %1, %2, %2, 0x96;" \
                     : "=r"(d64_mma_b0) \
                     : "r"(as_u32_ref(tSrK[N][K][CN][0])), \
                       "r"(__float_as_uint(softmax.row_max[DM][CM]))); \
        mma::m16n8k16_f32f16f16f32_accum( \
            tSrS[M][N][CN][0][0], tSrS[M][N][CN][0][1], \
            tSrS[M][N][CN][1][0], tSrS[M][N][CN][1][1], \
            as_u32_ref(tSrQ[M][K][0][0]), as_u32_ref(tSrQ[M][K][0][1]), \
            as_u32_ref(tSrQ[M][K][1][0]), as_u32_ref(tSrQ[M][K][1][1]), \
            d64_mma_b0, as_u32_ref(tSrK[N][K][CN][1])); \
      } while (0)
#define FAV2_D64_SCORE_SCALE(M, N, CN, CM, C) \
      tSrS[M][N][CN][CM][C] *= d64_score_log2e
#define FAV2_D64_SCORE_SCALE4(M, N, CN) \
      FAV2_D64_SCORE_SCALE(M, N, CN, 0, 0); \
      FAV2_D64_SCORE_SCALE(M, N, CN, 0, 1); \
      FAV2_D64_SCORE_SCALE(M, N, CN, 1, 0); \
      FAV2_D64_SCORE_SCALE(M, N, CN, 1, 1)
#define FAV2_D64_SCORE_SCALE_DEP(M, N, CN, CM, C, DM, DN, DCN, DCM, DC) \
      do { \
        uint32_t d64_score_bits; \
        asm volatile("lop3.b32 %0, %1, %2, %2, 0x96;" \
                     : "=r"(d64_score_bits) \
                     : "r"(__float_as_uint(tSrS[M][N][CN][CM][C])), \
                       "r"(__float_as_uint(tSrS[DM][DN][DCN][DCM][DC]))); \
        tSrS[M][N][CN][CM][C] = \
            __uint_as_float(d64_score_bits) * d64_score_log2e; \
      } while (0)
#define FAV2_D64_SCORE_SCALE_DEP4(M, N, CN, DM, DN, DCN, DCM, DC) \
      FAV2_D64_SCORE_SCALE_DEP(M, N, CN, 0, 0, DM, DN, DCN, DCM, DC); \
      FAV2_D64_SCORE_SCALE(M, N, CN, 0, 1); \
      FAV2_D64_SCORE_SCALE(M, N, CN, 1, 0); \
      FAV2_D64_SCORE_SCALE(M, N, CN, 1, 1)
#define FAV2_D64_SCORE_MAX(M, N, CN, CM, C) \
      softmax.row_max[M][CM] = fmaxf( \
          softmax.row_max[M][CM], tSrS[M][N][CN][CM][C])
#define FAV2_D64_SCORE_MAX4(M, N, CN) \
      FAV2_D64_SCORE_MAX(M, N, CN, 0, 0); \
      FAV2_D64_SCORE_MAX(M, N, CN, 0, 1); \
      FAV2_D64_SCORE_MAX(M, N, CN, 1, 0); \
      FAV2_D64_SCORE_MAX(M, N, CN, 1, 1)

      clearS(tSrS);

      // Load all fragments, then issue four small output-tile wavefronts.
      FAV2_D64_LOAD_K(0, 0); FAV2_D64_LOAD_K(1, 0);
      FAV2_D64_LOAD_K(2, 0); FAV2_D64_LOAD_K(3, 0);
      FAV2_D64_LOAD_K(0, 1); FAV2_D64_LOAD_K(1, 1);
      FAV2_D64_LOAD_K(2, 1); FAV2_D64_LOAD_K(3, 1);
      FAV2_D64_LOAD_K(0, 2); FAV2_D64_LOAD_K(1, 2);
      FAV2_D64_LOAD_K(2, 2); FAV2_D64_LOAD_K(3, 2);
      FAV2_D64_LOAD_K(0, 3); FAV2_D64_LOAD_K(1, 3);
      FAV2_D64_LOAD_K(2, 3); FAV2_D64_LOAD_K(3, 3);

      // Wavefront 0: K0/K1/K2 stay independent; K3 follows each score.
      FAV2_D64_MMA(0, 0, 0, 0);
      FAV2_D64_MMA(1, 0, 0, 0);
      FAV2_D64_MMA(0, 0, 1, 0);
      FAV2_D64_MMA(1, 0, 1, 0);
      FAV2_D64_MMA(0, 0, 0, 1);
      FAV2_D64_MMA(1, 0, 0, 1);
      FAV2_D64_MMA(0, 0, 1, 1);
      FAV2_D64_MMA(1, 0, 1, 1);
      FAV2_D64_MMA(0, 0, 0, 2);
      FAV2_D64_MMA(1, 0, 0, 2);
      FAV2_D64_MMA(0, 0, 1, 2);
      FAV2_D64_MMA(1, 0, 1, 2);
      FAV2_D64_MMA(0, 0, 0, 3);
      FAV2_D64_MMA(1, 0, 0, 3);
      FAV2_D64_SCORE_SCALE_DEP4(0, 0, 0, 1, 0, 0, 0, 0);
      FAV2_D64_SCORE_SCALE_DEP4(1, 0, 0, 0, 0, 0, 0, 0);
      FAV2_D64_MMA(0, 0, 1, 3);
      FAV2_D64_SCORE_SCALE_DEP4(0, 0, 1, 0, 1, 0, 0, 0);
      FAV2_D64_MMA(1, 0, 1, 3);
      FAV2_D64_SCORE_SCALE_DEP4(1, 0, 1, 0, 1, 0, 0, 0);

      // Wavefront 1: K0/K1/K2 stay independent; K3 follows each score.
      FAV2_D64_MMA(0, 1, 0, 0);
      FAV2_D64_MMA(1, 1, 0, 0);
      FAV2_D64_MMA(0, 1, 1, 0);
      FAV2_D64_MMA(1, 1, 1, 0);
      FAV2_D64_MMA(0, 1, 0, 1);
      FAV2_D64_MMA(1, 1, 0, 1);
      FAV2_D64_MMA(0, 1, 1, 1);
      FAV2_D64_MMA(1, 1, 1, 1);
      FAV2_D64_MMA(0, 1, 0, 2);
      FAV2_D64_MMA(1, 1, 0, 2);
      FAV2_D64_MMA(0, 1, 1, 2);
      FAV2_D64_MMA(1, 1, 1, 2);
      FAV2_D64_MMA(0, 1, 0, 3);
      FAV2_D64_MMA(1, 1, 0, 3);
      FAV2_D64_SCORE_SCALE4(0, 1, 0);
      FAV2_D64_SCORE_SCALE4(1, 1, 0);
      FAV2_D64_MMA(0, 1, 1, 3);
      FAV2_D64_SCORE_SCALE4(0, 1, 1);
      FAV2_D64_MMA(1, 1, 1, 3);
      FAV2_D64_SCORE_SCALE4(1, 1, 1);

      FAV2_D64_SCORE_MAX4(0, 0, 0);
      FAV2_D64_SCORE_MAX4(0, 0, 1);
      FAV2_D64_SCORE_MAX4(1, 0, 0);
      FAV2_D64_SCORE_MAX4(1, 0, 1);

      // Wavefront 2: K0/K1/K2 stay independent; K3 follows each score.
      FAV2_D64_MMA(0, 2, 0, 0);
      FAV2_D64_MMA(1, 2, 0, 0);
      FAV2_D64_MMA(0, 2, 1, 0);
      FAV2_D64_MMA(1, 2, 1, 0);
      FAV2_D64_MMA(0, 2, 0, 1);
      FAV2_D64_MMA(1, 2, 0, 1);
      FAV2_D64_MMA(0, 2, 1, 1);
      FAV2_D64_MMA(1, 2, 1, 1);
      FAV2_D64_MMA(0, 2, 0, 2);
      FAV2_D64_MMA(1, 2, 0, 2);
      FAV2_D64_MMA(0, 2, 1, 2);
      FAV2_D64_MMA(1, 2, 1, 2);
      FAV2_D64_MMA_AFTER_MAX(0, 2, 0, 3, 0, 0);
      FAV2_D64_MMA_AFTER_MAX(1, 2, 0, 3, 1, 0);
      FAV2_D64_SCORE_SCALE4(0, 2, 0);
      FAV2_D64_SCORE_SCALE4(1, 2, 0);
      FAV2_D64_MMA_AFTER_MAX(0, 2, 1, 3, 0, 1);
      FAV2_D64_SCORE_SCALE4(0, 2, 1);
      FAV2_D64_MMA_AFTER_MAX(1, 2, 1, 3, 1, 1);
      FAV2_D64_SCORE_SCALE4(1, 2, 1);

      FAV2_D64_SCORE_MAX4(0, 1, 0);
      FAV2_D64_SCORE_MAX4(0, 1, 1);
      FAV2_D64_SCORE_MAX4(1, 1, 0);
      FAV2_D64_SCORE_MAX4(1, 1, 1);

      // Wavefront 3: K0/K1/K2 stay independent; K3 follows each score.
      FAV2_D64_MMA(0, 3, 0, 0);
      FAV2_D64_MMA(1, 3, 0, 0);
      FAV2_D64_MMA(0, 3, 1, 0);
      FAV2_D64_MMA(1, 3, 1, 0);
      FAV2_D64_MMA(0, 3, 0, 1);
      FAV2_D64_MMA(1, 3, 0, 1);
      FAV2_D64_MMA(0, 3, 1, 1);
      FAV2_D64_MMA(1, 3, 1, 1);
      FAV2_D64_MMA(0, 3, 0, 2);
      FAV2_D64_MMA(1, 3, 0, 2);
      FAV2_D64_MMA(0, 3, 1, 2);
      FAV2_D64_MMA(1, 3, 1, 2);
      FAV2_D64_MMA_AFTER_MAX(0, 3, 0, 3, 0, 0);
      FAV2_D64_MMA_AFTER_MAX(1, 3, 0, 3, 1, 0);
      FAV2_D64_SCORE_SCALE4(0, 3, 0);
      FAV2_D64_SCORE_SCALE4(1, 3, 0);
      FAV2_D64_MMA_AFTER_MAX(0, 3, 1, 3, 0, 1);
      FAV2_D64_SCORE_SCALE4(0, 3, 1);
      FAV2_D64_MMA_AFTER_MAX(1, 3, 1, 3, 1, 1);
      FAV2_D64_SCORE_SCALE4(1, 3, 1);
      FAV2_D64_SCORE_MAX4(0, 2, 0);
      FAV2_D64_SCORE_MAX4(0, 2, 1);
      FAV2_D64_SCORE_MAX4(1, 2, 0);
      FAV2_D64_SCORE_MAX4(1, 2, 1);
      FAV2_D64_SCORE_MAX4(0, 3, 0);
      FAV2_D64_SCORE_MAX4(0, 3, 1);
      FAV2_D64_SCORE_MAX4(1, 3, 0);
      FAV2_D64_SCORE_MAX4(1, 3, 1);
      // Finish the row-max allreduce here, but keep the O-rescale factors
      // live for the PV schedule below.  The old helper also scaled every O
      // accumulator here, creating a long 64-FMUL block before the first PV
      // HMMA.  cuDNN distributes those FMULs across the first PV wavefront.
      softmax::MaxOp d64_max_op;
      softmax.row_max[0][0] =
          softmax::allreduce_4(softmax.row_max[0][0], d64_max_op);
      softmax.row_max[0][1] =
          softmax::allreduce_4(softmax.row_max[0][1], d64_max_op);
      softmax.row_max[1][0] =
          softmax::allreduce_4(softmax.row_max[1][0], d64_max_op);
      softmax.row_max[1][1] =
          softmax::allreduce_4(softmax.row_max[1][1], d64_max_op);
      if (k_tile == initial_k_tile) {
        d64_o_scale[0][0] = 1.0F;
        d64_o_scale[0][1] = 1.0F;
        d64_o_scale[1][0] = 1.0F;
        d64_o_scale[1][1] = 1.0F;
      } else {
        d64_o_scale[0][0] = exp2f(__fmaf_rn(
            d64_prev_max[0][0], 1.0F, -softmax.row_max[0][0]));
        d64_o_scale[0][1] = exp2f(__fmaf_rn(
            d64_prev_max[0][1], 1.0F, -softmax.row_max[0][1]));
        d64_o_scale[1][0] = exp2f(__fmaf_rn(
            d64_prev_max[1][0], 1.0F, -softmax.row_max[1][0]));
        d64_o_scale[1][1] = exp2f(__fmaf_rn(
            d64_prev_max[1][1], 1.0F, -softmax.row_max[1][1]));
        softmax.row_sum[0][0] *= d64_o_scale[0][0];
        softmax.row_sum[0][1] *= d64_o_scale[0][1];
        softmax.row_sum[1][0] *= d64_o_scale[1][0];
        softmax.row_sum[1][1] *= d64_o_scale[1][1];
      }

#undef FAV2_D64_SCORE_MAX4
#undef FAV2_D64_SCORE_MAX
#undef FAV2_D64_SCORE_SCALE4
#undef FAV2_D64_SCORE_SCALE_DEP4
#undef FAV2_D64_SCORE_SCALE_DEP
#undef FAV2_D64_SCORE_SCALE
#undef FAV2_D64_MMA_AFTER_MAX
#undef FAV2_D64_MMA
#undef FAV2_D64_LOAD_K
    } else {
#endif
      clearS(tSrS);
      compute_score_ss<kHeadDim, kBlockM, kBlockN, MMA_M, MMA_N, MMA_K>(
          sQ, sK, &tSrQ[0][0][0][0][0], &tSrK[0][0][0][0][0],
          &tSrS[0][0][0][0][0]);
#if FAV2_D64_ISSUE_SCHEDULE
    }
#endif

    // --------------------------------------------------------
    // Need V_current before P @ V.
    // --------------------------------------------------------
    cp_async::wait_group<0>();
    __syncthreads();

    // Current K is no longer needed after compute_score_ss().
    // So we can overwrite sK with K_prev.
    if (k_tile > 0) {
      const half *gK_prev = gK_base + (k_tile - 1) * kBlockN * KeyRowStride;

      async_load_x_tensor<kHeadDim, kQKVSmemStride, kBlockN, kThreads>(
          sK, gK_prev, KeyRowStride);
    }

    // --------------------------------------------------------
    // softmax online + acc_o += P @ V_current
    // Uses sV.
    // K_prev load can overlap with this compute.
    // --------------------------------------------------------
#if FAV2_D64_ISSUE_SCHEDULE
    if constexpr (kHeadDim == 64 && kBlockN == 64) {
      const int d64_lane_id = threadIdx.x % 32;
      const int d64_tOsV_row = d64_lane_id % 16;
      const int d64_tOsV_col = (d64_lane_id / 16) * 8;

      if (k_tile == initial_k_tile) {
        softmax.row_sum[0][0] = 0.0F;
        softmax.row_sum[0][1] = 0.0F;
        softmax.row_sum[1][0] = 0.0F;
        softmax.row_sum[1][1] = 0.0F;
      }

#define FAV2_D64_EXP_PACK(M, N, CN, CM) \
      tSrS[M][N][CN][CM][0] = exp2f(__fmaf_rn( \
          tSrS[M][N][CN][CM][0], 1.0F, -softmax.row_max[M][CM])); \
      tSrS[M][N][CN][CM][1] = exp2f(__fmaf_rn( \
          tSrS[M][N][CN][CM][1], 1.0F, -softmax.row_max[M][CM])); \
      as_u32_ref(tOrP[M][N][CN][CM]) = \
          fav2_pack_f32x2_to_f16x2(tSrS[M][N][CN][CM][0], \
                                   tSrS[M][N][CN][CM][1])
#define FAV2_D64_EXP_PAIR(N, M, CN) \
      FAV2_D64_EXP_PACK(M, N, CN, 0); \
      FAV2_D64_EXP_PACK(M, N, CN, 1)
#define FAV2_D64_SUM_PAIR0(M, CN) \
      softmax.row_sum[M][0] += tSrS[M][0][CN][0][0]; \
      softmax.row_sum[M][0] += tSrS[M][0][CN][0][1]; \
      softmax.row_sum[M][1] += tSrS[M][0][CN][1][0]; \
      softmax.row_sum[M][1] += tSrS[M][0][CN][1][1]
#define FAV2_D64_SUM_PARTIAL0(N, M) \
      tSrS[M][N][0][0][0] += tSrS[M][N][0][0][1]; \
      tSrS[M][N][0][1][0] += tSrS[M][N][0][1][1]
#define FAV2_D64_SUM_PARTIAL1(N, M) \
      tSrS[M][N][0][0][0] += tSrS[M][N][1][0][0]; \
      tSrS[M][N][0][0][0] += tSrS[M][N][1][0][1]; \
      tSrS[M][N][0][1][0] += tSrS[M][N][1][1][0]; \
      tSrS[M][N][0][1][0] += tSrS[M][N][1][1][1]
#define FAV2_D64_MERGE_SUM(N) \
      softmax.row_sum[0][0] += tSrS[0][N][0][0][0]; \
      softmax.row_sum[0][1] += tSrS[0][N][0][1][0]; \
      softmax.row_sum[1][0] += tSrS[1][N][0][0][0]; \
      softmax.row_sum[1][1] += tSrS[1][N][0][1][0]
#define FAV2_D64_LOAD_V(K, N) \
      ldsm::x4<ldsm::T>( \
          tOrV[K][N][0][0], tOrV[K][N][0][1], tOrV[K][N][1][0], \
          tOrV[K][N][1][1], \
          fav2_padded_smem_ptr<kQKVSmemStride>( \
              sV, d64_tOsV_row + N * 16, \
              K * (kHeadDim / MMA_K) + d64_tOsV_col))
#define FAV2_D64_PV(M, K, CK, N) \
      mma::m16n8k16_f32f16f16f32_accum( \
          tOrO[M][K][CK][0][0], tOrO[M][K][CK][0][1], \
          tOrO[M][K][CK][1][0], tOrO[M][K][CK][1][1], \
          tOrP[M][N][0][0], tOrP[M][N][0][1], \
          tOrP[M][N][1][0], tOrP[M][N][1][1], \
          tOrV[K][N][CK][0], tOrV[K][N][CK][1])
#define FAV2_D64_RESCALE_O4(M, K, CK) \
      tOrO[M][K][CK][0][0] *= d64_o_scale[M][0]; \
      tOrO[M][K][CK][0][1] *= d64_o_scale[M][0]; \
      tOrO[M][K][CK][1][0] *= d64_o_scale[M][1]; \
      tOrO[M][K][CK][1][1] *= d64_o_scale[M][1]
#define FAV2_D64_PV_FIRST(M, K, CK, N) \
      do { \
        if (k_tile != initial_k_tile) { \
          FAV2_D64_RESCALE_O4(M, K, CK); \
        } \
        FAV2_D64_PV(M, K, CK, N); \
      } while (0)

      // N=0 is made ready first.  While its PV HMMA train runs, exp/pack
      // for N=1 is issued into the scalar pipes.
      FAV2_D64_LOAD_V(0, 0); FAV2_D64_LOAD_V(1, 0);
      FAV2_D64_LOAD_V(2, 0); FAV2_D64_LOAD_V(3, 0);
      FAV2_D64_EXP_PAIR(0, 0, 0); FAV2_D64_EXP_PAIR(0, 0, 1);
      FAV2_D64_EXP_PAIR(0, 1, 0); FAV2_D64_EXP_PAIR(0, 1, 1);

      FAV2_D64_LOAD_V(0, 1); FAV2_D64_LOAD_V(1, 1);
      FAV2_D64_LOAD_V(2, 1); FAV2_D64_LOAD_V(3, 1);
      FAV2_D64_PV_FIRST(0, 0, 0, 0); FAV2_D64_SUM_PAIR0(0, 0); FAV2_D64_EXP_PAIR(1, 0, 0);
      FAV2_D64_PV_FIRST(0, 0, 1, 0); FAV2_D64_SUM_PAIR0(0, 1); FAV2_D64_EXP_PAIR(1, 0, 1);
      FAV2_D64_PV_FIRST(0, 1, 0, 0); FAV2_D64_SUM_PAIR0(1, 0); FAV2_D64_EXP_PAIR(1, 1, 0);
      FAV2_D64_PV_FIRST(0, 1, 1, 0); FAV2_D64_SUM_PAIR0(1, 1); FAV2_D64_EXP_PAIR(1, 1, 1);
      FAV2_D64_PV_FIRST(0, 2, 0, 0); FAV2_D64_SUM_PARTIAL0(1, 0);
      FAV2_D64_PV_FIRST(0, 2, 1, 0); FAV2_D64_SUM_PARTIAL1(1, 0);
      FAV2_D64_PV_FIRST(0, 3, 0, 0); FAV2_D64_SUM_PARTIAL0(1, 1);
      FAV2_D64_PV_FIRST(0, 3, 1, 0); FAV2_D64_SUM_PARTIAL1(1, 1);
      FAV2_D64_MERGE_SUM(1);
      FAV2_D64_PV_FIRST(1, 0, 0, 0);
      FAV2_D64_PV_FIRST(1, 0, 1, 0);
      FAV2_D64_PV_FIRST(1, 1, 0, 0);
      FAV2_D64_PV_FIRST(1, 1, 1, 0);
      FAV2_D64_PV_FIRST(1, 2, 0, 0);
      FAV2_D64_PV_FIRST(1, 2, 1, 0);
      FAV2_D64_PV_FIRST(1, 3, 0, 0);
      FAV2_D64_PV_FIRST(1, 3, 1, 0);

      // Repeat the same software pipeline for N=1 -> N=2.
      FAV2_D64_LOAD_V(0, 2); FAV2_D64_LOAD_V(1, 2);
      FAV2_D64_LOAD_V(2, 2); FAV2_D64_LOAD_V(3, 2);
      FAV2_D64_PV(0, 0, 0, 1); FAV2_D64_EXP_PAIR(2, 0, 0);
      FAV2_D64_PV(0, 0, 1, 1); FAV2_D64_SUM_PARTIAL0(2, 0); FAV2_D64_EXP_PAIR(2, 0, 1);
      FAV2_D64_PV(0, 1, 0, 1); FAV2_D64_SUM_PARTIAL1(2, 0); FAV2_D64_EXP_PAIR(2, 1, 0);
      FAV2_D64_PV(0, 1, 1, 1); FAV2_D64_SUM_PARTIAL0(2, 1); FAV2_D64_EXP_PAIR(2, 1, 1);
      FAV2_D64_PV(0, 2, 0, 1); FAV2_D64_SUM_PARTIAL1(2, 1);
      FAV2_D64_MERGE_SUM(2);
      FAV2_D64_PV(0, 2, 1, 1);
      FAV2_D64_PV(0, 3, 0, 1);
      FAV2_D64_PV(0, 3, 1, 1);
      FAV2_D64_PV(1, 0, 0, 1);
      FAV2_D64_PV(1, 0, 1, 1);
      FAV2_D64_PV(1, 1, 0, 1);
      FAV2_D64_PV(1, 1, 1, 1);
      FAV2_D64_PV(1, 2, 0, 1);
      FAV2_D64_PV(1, 2, 1, 1);
      FAV2_D64_PV(1, 3, 0, 1);
      FAV2_D64_PV(1, 3, 1, 1);

      // N=2 -> N=3.
      FAV2_D64_LOAD_V(0, 3); FAV2_D64_LOAD_V(1, 3);
      FAV2_D64_LOAD_V(2, 3); FAV2_D64_LOAD_V(3, 3);
      FAV2_D64_PV(0, 0, 0, 2); FAV2_D64_EXP_PAIR(3, 0, 0);
      FAV2_D64_PV(0, 0, 1, 2); FAV2_D64_SUM_PARTIAL0(3, 0); FAV2_D64_EXP_PAIR(3, 0, 1);
      FAV2_D64_PV(0, 1, 0, 2); FAV2_D64_SUM_PARTIAL1(3, 0); FAV2_D64_EXP_PAIR(3, 1, 0);
      FAV2_D64_PV(0, 1, 1, 2); FAV2_D64_SUM_PARTIAL0(3, 1); FAV2_D64_EXP_PAIR(3, 1, 1);
      FAV2_D64_PV(0, 2, 0, 2); FAV2_D64_SUM_PARTIAL1(3, 1);
      FAV2_D64_MERGE_SUM(3);
      FAV2_D64_PV(0, 2, 1, 2);
      FAV2_D64_PV(0, 3, 0, 2);
      FAV2_D64_PV(0, 3, 1, 2);
      FAV2_D64_PV(1, 0, 0, 2);
      FAV2_D64_PV(1, 0, 1, 2);
      FAV2_D64_PV(1, 1, 0, 2);
      FAV2_D64_PV(1, 1, 1, 2);
      FAV2_D64_PV(1, 2, 0, 2);
      FAV2_D64_PV(1, 2, 1, 2);
      FAV2_D64_PV(1, 3, 0, 2);
      FAV2_D64_PV(1, 3, 1, 2);

      // N=3 has no following score tile, so finish its PV chain directly.
      FAV2_D64_PV(0, 0, 0, 3); FAV2_D64_PV(0, 0, 1, 3);
      FAV2_D64_PV(0, 1, 0, 3); FAV2_D64_PV(0, 1, 1, 3);
      FAV2_D64_PV(0, 2, 0, 3); FAV2_D64_PV(0, 2, 1, 3);
      FAV2_D64_PV(0, 3, 0, 3); FAV2_D64_PV(0, 3, 1, 3);
      FAV2_D64_PV(1, 0, 0, 3); FAV2_D64_PV(1, 0, 1, 3);
      FAV2_D64_PV(1, 1, 0, 3); FAV2_D64_PV(1, 1, 1, 3);
      FAV2_D64_PV(1, 2, 0, 3); FAV2_D64_PV(1, 2, 1, 3);
      FAV2_D64_PV(1, 3, 0, 3); FAV2_D64_PV(1, 3, 1, 3);

#undef FAV2_D64_PV
#undef FAV2_D64_PV_FIRST
#undef FAV2_D64_RESCALE_O4
#undef FAV2_D64_LOAD_V
#undef FAV2_D64_EXP_PAIR
#undef FAV2_D64_SUM_PAIR0
#undef FAV2_D64_SUM_PARTIAL0
#undef FAV2_D64_SUM_PARTIAL1
#undef FAV2_D64_MERGE_SUM
#undef FAV2_D64_EXP_PACK
    } else {
#endif
      if (k_tile == initial_k_tile) {
        softmax.softmax_rescale_o<true>(
            tSrS, tOrO, Strides::softmax_scale_log2(params));
      } else {
        softmax.softmax_rescale_o<false>(
            tSrS, tOrO, Strides::softmax_scale_log2(params));
      }
      convertS(tSrS, tOrP);
      compute_output_rs<kHeadDim, kBlockM, kBlockN, MMA_M, MMA_N, MMA_K>(
          &tOrP[0][0][0][0][0], sV, &tOrV[0][0][0][0][0],
          &tOrO[0][0][0][0][0]);
#if FAV2_D64_ISSUE_SCHEDULE
    }
#endif

  }

  softmax.normalize(tOrO);
  store_output_epilogue<kHeadDim, kBlockM, kWarps, kThreads, MMA_M, MMA_K>(
      sQ, gO, OutputRowStride, tOrO);
}

template <int kHeadDim, int kBlockM, int kBlockN,
          typename Strides = Fav2RuntimeStrides,
          typename Params = FlashFwdParams<kHeadDim>>
__global__ void flash_attn_v2(Params params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;

  compute_attn_1rowblock<kHeadDim, kBlockM, kBlockN, Strides, Params>(
      params, bidb, bidh, m_block);
}

template <typename Config>
__global__ void flash_attn_v2_static(Fav2StaticKernelPtrs params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;

  compute_attn_1rowblock<Config::kHeadDim, Config::kBlockM, Config::kBlockN,
                         Config, Fav2StaticKernelPtrs>(params, bidb, bidh,
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
  static constexpr int kQKVSmemElements =
      (kBlockM + 2 * kBlockN) * (kHeadDim + kFav2SmemPad);
  static constexpr int kOSmemElements =
      kBlockM * (kHeadDim + kFav2SmemPad);
  static constexpr int kSmemBytes =
      (kQKVSmemElements > kOSmemElements ? kQKVSmemElements
                                         : kOSmemElements) *
      sizeof(half);
};

template <int kHeadDim> struct FlashAttnV2A100Config {
  static constexpr bool kSupported = false;
  static constexpr int kBlockMValue = 0;
  static constexpr int kBlockNValue = 0;
};


template <> struct FlashAttnV2A100Config<32>
    : FlashAttnV2LaunchConfig<32, 128, 128> {
  static constexpr bool kSupported = true;
};

template <> struct FlashAttnV2A100Config<64>
    : FlashAttnV2LaunchConfig<64, 128, 64> {
  static constexpr bool kSupported = true;
};

template <> struct FlashAttnV2A100Config<128>
    : FlashAttnV2LaunchConfig<128, 128, 64> {
  static constexpr bool kSupported = true;
};

template <int kHeadDim, int kBlockM, int kBlockN,
          typename Strides = Fav2RuntimeStrides>
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

  auto kernel_fptr = flash_attn_v2<kHeadDim, kBlockM, kBlockN, Strides>;

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

inline cudaError_t launch_flash_attn_v2_static_b2_sq4096_sk4096_h8_d64(
    Fav2StaticKernelPtrs params, cudaStream_t stream = 0) {
  using Config = Fav2StaticB2Sq4096Sk4096H8D64Config;
  using LaunchConfig =
      FlashAttnV2LaunchConfig<Config::kHeadDim, Config::kBlockM,
                             Config::kBlockN>;
  auto kernel_fptr = flash_attn_v2_static<Config>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      LaunchConfig::kSmemBytes);
  if (err != cudaSuccess) {
    return err;
  }

  dim3 block(LaunchConfig::kThreads);
  dim3 grid(Config::kSeqlenQ / LaunchConfig::kBlockMValue,
            Config::kBatchSize, Config::kHeadsQ);
  kernel_fptr<<<grid, block, LaunchConfig::kSmemBytes, stream>>>(params);
  return cudaGetLastError();
}

inline cudaError_t launch_flash_attn_v2_static_b1_sq4096_sk4032_h16_d128(
    Fav2StaticKernelPtrs params, cudaStream_t stream = 0) {
  using Config = Fav2StaticB1Sq4096Sk4032H16D128Config;
  using LaunchConfig =
      FlashAttnV2LaunchConfig<Config::kHeadDim, Config::kBlockM,
                             Config::kBlockN>;
  auto kernel_fptr = flash_attn_v2_static<Config>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      LaunchConfig::kSmemBytes);
  if (err != cudaSuccess) {
    return err;
  }

  dim3 block(LaunchConfig::kThreads);
  dim3 grid(Config::kSeqlenQ / LaunchConfig::kBlockMValue,
            Config::kBatchSize, Config::kHeadsQ);
  kernel_fptr<<<grid, block, LaunchConfig::kSmemBytes, stream>>>(params);
  return cudaGetLastError();
}

} // namespace fav2_sm80

} // namespace felix::detail::sm80_flash_attn_f16_hd64_best
