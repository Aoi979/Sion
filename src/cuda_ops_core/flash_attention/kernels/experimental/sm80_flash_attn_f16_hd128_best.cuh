#pragma once
#include "../../detail/sm80_flash_attn_f16_hd128_best_trait.cuh"
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Experimental source snapshot specialized for the D128 benchmark target.
namespace cuda_ops_core::detail::sm80_flash_attn_f16_hd128_best {

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

template <int MMA_M, int MMA_N> struct Softmax {
  float row_max[MMA_M][2];
  float row_sum[MMA_M][2];

  __device__ __forceinline__ Softmax() {}

  template <bool IsFirst, int MMA_K>
  __device__ __forceinline__ void prepare_scaled_scores(
      const float (&row_max_prev)[MMA_M][2],
      float (&acc_o)[MMA_M][MMA_K][2][2][2]) {
#pragma unroll
    for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
      for (int cm = 0; cm < 2; ++cm) {
        row_max[m][cm] = allreduce_4(row_max[m][cm], MaxOp{});
      }
    }

    if constexpr (IsFirst) {
#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_sum[m][cm] = 0.0f;
        }
      }
    } else {
#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          const float scores_scale =
              exp2f(row_max_prev[m][cm] - row_max[m][cm]);
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

  template <int N>
  __device__ __forceinline__ void exp2_sum_n(
      float (&tSrS)[MMA_M][MMA_N][2][2][2]) {
#pragma unroll
    for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
      for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            tSrS[m][N][cn][cm][c] = exp2f(__fmaf_rn(
                tSrS[m][N][cn][cm][c], 1.0F, -row_max[m][cm]));
            row_sum[m][cm] += tSrS[m][N][cn][cm][c];
          }
        }
      }
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
  static constexpr float kSoftmaxScaleLog2 = 0.12751743082459868F;

};

// [B, S, H, D] contiguous target:
// B=1, Sq=4096, Sk=4032, Hq=Hk=16, D=128. Strides are in half elements.
using Fav2StaticB1Sq4096Sk4032H16D128Config =
    Fav2StaticConfig<128, 128, 64, 1, 4096, 4032, 16, 16, 1, 8388608,
                     128, 524288, 8257536, 128, 516096, 8257536, 128,
                     516096, 8388608, 128, 524288>;

// CudaOpsCore exposes a single sequence length for Q/K/V.  Keep the D128 body
// fixed to the project's contiguous [B, H, S, D] target rather than silently
// launching the standalone snapshot's separate-Sk=4032 target.
using Fav2ProjectB1Sq4096Sk4096H16D128Config =
    Fav2StaticConfig<128, 128, 64, 1, 4096, 4096, 16, 16, 1, 8388608,
                     128, 524288, 8388608, 128, 524288, 8388608, 128,
                     524288, 8388608, 128, 524288>;

// All scalar metadata is supplied by the template config.  Only data
// addresses remain as runtime kernel arguments.
struct Fav2StaticKernelPtrs {
  const half *q;
  const half *k;
  const half *v;
  half *o;
};

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

// D128-only structural path. Q/K/V fragments are tile-local; only the
// score/output accumulators span a KV iteration.
template <int K, bool Final, int kBlockM, int kBlockN>
__device__ __forceinline__ void compute_score_ss_d128_k(
    half *sQ, half *sK, float (&tSrS)[2][4][2][2][2],
    float (&row_max)[2][2], float softmax_scale_log2) {
  static_assert(kBlockM == 128);
  static_assert(kBlockN == 64);

  constexpr int kQSmemStride = 128 + kFav2SmemPad;
  constexpr int kKSmemStride = 128 + kFav2SmemPad;
  const int lane_id = threadIdx.x % 32;
  const int tSsQ_row = lane_id % 16;
  const int tSsQ_col = lane_id / 16;
  const int tSsK_row = (lane_id / 16) * 8 + lane_id % 8;
  const int tSsK_col = (lane_id % 16) / 8;
  const int warp_id = threadIdx.x / 32;

  // One K fragment is a complete lexical scope.  K is a template argument,
  // so both the address and the register lifetime are visible as constants to
  // ptxas, exactly like a hand-written SASS sequence.
  half q_tile[2][2][2][2];
  half k_tile[4][2][2][2];
  ldsm::x4<ldsm::N>(
      q_tile[0][0][0], q_tile[0][0][1], q_tile[0][1][0], q_tile[0][1][1],
      fav2_padded_smem_ptr<kQSmemStride>(
          sQ, warp_id * 16 + tSsQ_row, tSsQ_col * 8 + K * 16));
  ldsm::x4<ldsm::N>(
      q_tile[1][0][0], q_tile[1][0][1], q_tile[1][1][0], q_tile[1][1][1],
      fav2_padded_smem_ptr<kQSmemStride>(
          sQ, kBlockM / 2 + warp_id * 16 + tSsQ_row,
          tSsQ_col * 8 + K * 16));
  ldsm::x4<ldsm::N>(
      k_tile[0][0][0], k_tile[0][0][1], k_tile[0][1][0], k_tile[0][1][1],
      fav2_padded_smem_ptr<kKSmemStride>(
          sK, tSsK_row, tSsK_col * 8 + K * 16));
  ldsm::x4<ldsm::N>(
      k_tile[1][0][0], k_tile[1][0][1], k_tile[1][1][0], k_tile[1][1][1],
      fav2_padded_smem_ptr<kKSmemStride>(
          sK, kBlockN / 4 + tSsK_row, tSsK_col * 8 + K * 16));
  ldsm::x4<ldsm::N>(
      k_tile[2][0][0], k_tile[2][0][1], k_tile[2][1][0], k_tile[2][1][1],
      fav2_padded_smem_ptr<kKSmemStride>(
          sK, kBlockN / 2 + tSsK_row, tSsK_col * 8 + K * 16));
  ldsm::x4<ldsm::N>(
      k_tile[3][0][0], k_tile[3][0][1], k_tile[3][1][0], k_tile[3][1][1],
      fav2_padded_smem_ptr<kKSmemStride>(
          sK, 3 * (kBlockN / 4) + tSsK_row, tSsK_col * 8 + K * 16));

#define FAV2_D128_SCORE_POST(CK, M, N)                                  \
  if constexpr (Final) {                                                 \
    tSrS[M][N][CK][0][0] *= softmax_scale_log2;                           \
    row_max[M][0] = fmaxf(row_max[M][0], tSrS[M][N][CK][0][0]);           \
    tSrS[M][N][CK][0][1] *= softmax_scale_log2;                           \
    row_max[M][0] = fmaxf(row_max[M][0], tSrS[M][N][CK][0][1]);           \
    tSrS[M][N][CK][1][0] *= softmax_scale_log2;                           \
    row_max[M][1] = fmaxf(row_max[M][1], tSrS[M][N][CK][1][0]);           \
    tSrS[M][N][CK][1][1] *= softmax_scale_log2;                           \
    row_max[M][1] = fmaxf(row_max[M][1], tSrS[M][N][CK][1][1]);           \
  }

#define FAV2_D128_SCORE_MMA(M, N)                                       \
  do {                                                                   \
    mma::m16n8k16_f32f16f16f32_accum(                                   \
        tSrS[M][N][0][0][0], tSrS[M][N][0][0][1],                       \
        tSrS[M][N][0][1][0], tSrS[M][N][0][1][1],                       \
        q_tile[M][0][0], q_tile[M][0][1],                               \
        q_tile[M][1][0], q_tile[M][1][1],                               \
        k_tile[N][0][0], k_tile[N][0][1]);                              \
    FAV2_D128_SCORE_POST(0, M, N);                                       \
    mma::m16n8k16_f32f16f16f32_accum(                                   \
        tSrS[M][N][1][0][0], tSrS[M][N][1][0][1],                       \
        tSrS[M][N][1][1][0], tSrS[M][N][1][1][1],                       \
        q_tile[M][0][0], q_tile[M][0][1],                               \
        q_tile[M][1][0], q_tile[M][1][1],                               \
        k_tile[N][1][0], k_tile[N][1][1]);                              \
    FAV2_D128_SCORE_POST(1, M, N);                                       \
  } while (0)

  FAV2_D128_SCORE_MMA(0, 0);
  FAV2_D128_SCORE_MMA(1, 0);
  FAV2_D128_SCORE_MMA(1, 1);
  FAV2_D128_SCORE_MMA(0, 1);
  FAV2_D128_SCORE_MMA(0, 2);
  FAV2_D128_SCORE_MMA(1, 2);
  FAV2_D128_SCORE_MMA(1, 3);
  FAV2_D128_SCORE_MMA(0, 3);

#undef FAV2_D128_SCORE_MMA
#undef FAV2_D128_SCORE_POST
}

template <int kBlockM, int kBlockN>
__device__ __forceinline__ void compute_score_ss_d128_tiled(
    half *sQ, half *sK,
    float (&tSrS)[2][4][2][2][2], float (&row_max)[2][2],
    float softmax_scale_log2) {
  // Keep the K-tile sequence explicit.  Only K=7 is allowed to scale and
  // reduce scores; the preceding seven calls are pure HMMA/LDSM waves.
  compute_score_ss_d128_k<0, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<1, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<2, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<3, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<4, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<5, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<6, false, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
  compute_score_ss_d128_k<7, true, kBlockM, kBlockN>(
      sQ, sK, tSrS, row_max, softmax_scale_log2);
}

template <int N>
__device__ __forceinline__ void load_v_d128_n(
    half (&v_n)[8][2][2][2], half *sV) {
  constexpr int kVSmemStride = 128 + kFav2SmemPad;
  const int lane_id = threadIdx.x % 32;
  const int tOsV_row = lane_id % 16;
  const int tOsV_col = (lane_id / 16) * 8;

#pragma unroll
  for (int k = 0; k < 8; ++k) {
    ldsm::x4<ldsm::T>(
        v_n[k][0][0], v_n[k][0][1], v_n[k][1][0], v_n[k][1][1],
        fav2_padded_smem_ptr<kVSmemStride>(
            sV, tOsV_row + N * 16, k * 16 + tOsV_col));
  }
}

__device__ __forceinline__ void compute_output_rs_d128_n(
    half (&tOrP_n)[2][2][2][2], half (&v_n)[8][2][2][2],
    float (&tOrO)[2][8][2][2][2]) {
  constexpr int kMmaM = 2;
  constexpr int kMmaK = 8;

#pragma unroll
  for (int m = 0; m < kMmaM; ++m) {
#pragma unroll
    for (int k = 0; k < kMmaK; ++k) {
      mma::m16n8k16_f32f16f16f32_accum(
          tOrO[m][k][0][0][0], tOrO[m][k][0][0][1],
          tOrO[m][k][0][1][0], tOrO[m][k][0][1][1],
          tOrP_n[m][0][0], tOrP_n[m][0][1], tOrP_n[m][1][0],
          tOrP_n[m][1][1], v_n[k][0][0], v_n[k][0][1]);
      mma::m16n8k16_f32f16f16f32_accum(
          tOrO[m][k][1][0][0], tOrO[m][k][1][0][1],
          tOrO[m][k][1][1][0], tOrO[m][k][1][1][1],
          tOrP_n[m][0][0], tOrP_n[m][0][1], tOrP_n[m][1][0],
          tOrP_n[m][1][1], v_n[k][1][0], v_n[k][1][1]);
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

template <int MMA_M, int N>
__device__ __forceinline__ void
convertS_n(float const (&tSrS)[MMA_M][4][2][2][2],
           half (&tOrP_n)[MMA_M][2][2][2]) {
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
      for (int cm = 0; cm < 2; ++cm) {
        const __half2 p =
            __floats2half2_rn(tSrS[m][N][cn][cm][0],
                              tSrS[m][N][cn][cm][1]);
        as_u32_ref(tOrP_n[m][cn][cm]) =
            *reinterpret_cast<const uint32_t *>(&p);
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

// D128-only entry point. All launch metadata is compile-time static.
template <typename Config>
__device__ __forceinline__ void compute_attn_1rowblock_d128_sonny(
    const Fav2StaticKernelPtrs &params, const int bidb, const int bidh,
    const int m_block) {
  static_assert(Config::kHeadDim == 128);
  static_assert(Config::kBlockM == 128);
  static_assert(Config::kBlockN == 64);

  constexpr int kWarps = 4;
  constexpr int kThreads = kWarps * 32;
  constexpr int kMmaM = 2;
  constexpr int kMmaN = 4;
  constexpr int kMmaK = 8;
  constexpr int kQKVSmemStride = 128 + kFav2SmemPad;

  const int kv_head = bidh / Config::kQHeadsPerKVHead;

  const int QueryBatchStride = Config::q_batch(params);
  const int QueryRowStride = Config::q_row(params);
  const int QueryHeadStride = Config::q_head(params);
  const int KeyBatchStride = Config::k_batch(params);
  const int KeyRowStride = Config::k_row(params);
  const int KeyHeadStride = Config::k_head(params);
  const int ValueBatchStride = Config::v_batch(params);
  const int ValueRowStride = Config::v_row(params);
  const int ValueHeadStride = Config::v_head(params);
  const int OutputBatchStride = Config::o_batch(params);
  const int OutputRowStride = Config::o_row(params);
  const int OutputHeadStride = Config::o_head(params);

  auto gQ = params.q + bidb * QueryBatchStride + bidh * QueryHeadStride +
            m_block * Config::kBlockM * QueryRowStride;
  auto gK_base = params.k + bidb * KeyBatchStride + kv_head * KeyHeadStride;
  auto gV_base = params.v + bidb * ValueBatchStride + kv_head * ValueHeadStride;
  auto gO = params.o + bidb * OutputBatchStride + bidh * OutputHeadStride +
            m_block * Config::kBlockM * OutputRowStride;

  extern __shared__ half smem[];
  half *sQ = smem;
  half *sK = sQ + Config::kBlockM * kQKVSmemStride;
  half *sV = sK + Config::kBlockN * kQKVSmemStride;

  // Only the score/probability/output accumulators span a KV iteration.  Q,
  // K and V fragments are allocated inside their K-tile helpers and die
  // before the next tile is materialized.
  float tSrS[kMmaM][kMmaN][2][2][2];
  float tOrO[kMmaM][kMmaK][2][2][2];
  clearO(tOrO);

  const int num_k_tiles = Config::kSeqlenK / Config::kBlockN;
  const int initial_k_tile = num_k_tiles - 1;

  async_load_x_tensor<Config::kHeadDim, kQKVSmemStride, Config::kBlockM,
                      kThreads>(sQ, gQ, QueryRowStride);
  async_load_x_tensor<Config::kHeadDim, kQKVSmemStride, Config::kBlockN,
                      kThreads>(
      sK, gK_base + initial_k_tile * Config::kBlockN * KeyRowStride,
      KeyRowStride);

  cp_async::wait_group<0>();
  __syncthreads();

  softmax::Softmax<kMmaM, kMmaN> softmax;
  float row_max_prev[kMmaM][2];
  half v_rf0[kMmaK][2][2][2];
  half v_rf1[kMmaK][2][2][2];

  for (int k_tile = initial_k_tile; k_tile >= 0; --k_tile) {
    if (k_tile != initial_k_tile) {
      cp_async::wait_group<0>();
      __syncthreads();
    }

    const half *gV =
        gV_base + k_tile * Config::kBlockN * ValueRowStride;
    async_load_x_tensor<Config::kHeadDim, kQKVSmemStride, Config::kBlockN,
                        kThreads>(sV, gV, ValueRowStride);

    clearS(tSrS);
    if (k_tile == initial_k_tile) {
#pragma unroll
      for (int m = 0; m < kMmaM; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          softmax.row_max[m][cm] = -CUDART_INF_F;
        }
      }
    } else {
#pragma unroll
      for (int m = 0; m < kMmaM; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          row_max_prev[m][cm] = softmax.row_max[m][cm];
          softmax.row_max[m][cm] = -CUDART_INF_F;
        }
      }
    }
    compute_score_ss_d128_tiled<Config::kBlockM, Config::kBlockN>(sQ, sK,
                                                                   tSrS,
                                                                   softmax.row_max,
                                                                   Config::kSoftmaxScaleLog2);

    cp_async::wait_group<0>();
    __syncthreads();

    if (k_tile > 0) {
      const half *gK_prev =
          gK_base + (k_tile - 1) * Config::kBlockN * KeyRowStride;
      async_load_x_tensor<Config::kHeadDim, kQKVSmemStride, Config::kBlockN,
                          kThreads>(sK, gK_prev, KeyRowStride);
    }

    if (k_tile == initial_k_tile) {
      softmax.prepare_scaled_scores<true>(row_max_prev, tOrO);
    } else {
      softmax.prepare_scaled_scores<false>(row_max_prev, tOrO);
    }

    // PV RF pipeline prologue: materialize V[0] before the first P tile.
    load_v_d128_n<0>(v_rf0, sV);
    {
      half tOrP_n[kMmaM][2][2][2];
      softmax.template exp2_sum_n<0>(tSrS);
      convertS_n<kMmaM, 0>(tSrS, tOrP_n);
      load_v_d128_n<1>(v_rf1, sV);
      compute_output_rs_d128_n(tOrP_n, v_rf0, tOrO);
    }
    {
      half tOrP_n[kMmaM][2][2][2];
      softmax.template exp2_sum_n<1>(tSrS);
      convertS_n<kMmaM, 1>(tSrS, tOrP_n);
      load_v_d128_n<2>(v_rf0, sV);
      compute_output_rs_d128_n(tOrP_n, v_rf1, tOrO);
    }
    {
      half tOrP_n[kMmaM][2][2][2];
      softmax.template exp2_sum_n<2>(tSrS);
      convertS_n<kMmaM, 2>(tSrS, tOrP_n);
      load_v_d128_n<3>(v_rf1, sV);
      compute_output_rs_d128_n(tOrP_n, v_rf0, tOrO);
    }
    {
      half tOrP_n[kMmaM][2][2][2];
      softmax.template exp2_sum_n<3>(tSrS);
      convertS_n<kMmaM, 3>(tSrS, tOrP_n);
      compute_output_rs_d128_n(tOrP_n, v_rf1, tOrO);
    }
  }

  softmax.normalize(tOrO);
  store_output_epilogue<Config::kHeadDim, Config::kBlockM, kWarps, kThreads,
                        kMmaM, kMmaK>(sQ, gO, OutputRowStride, tOrO);
}

template <typename Config>
__global__ void flash_attn_v2_static_d128_sonny(Fav2StaticKernelPtrs params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;

  compute_attn_1rowblock_d128_sonny<Config>(params, bidb, bidh, m_block);
}

namespace fav2_sm80 {

struct FlashAttnV2D128LaunchConfig {
  static constexpr int kBlockMValue = 128;
  static constexpr int kBlockNValue = 64;
  static constexpr int kWarps = 4;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kQKVSmemElements =
      (kBlockMValue + 2 * kBlockNValue) * (128 + kFav2SmemPad);
  static constexpr int kOSmemElements =
      kBlockMValue * (128 + kFav2SmemPad);
  static constexpr int kSmemBytes =
      (kQKVSmemElements > kOSmemElements ? kQKVSmemElements
                                         : kOSmemElements) *
      sizeof(half);
};
inline cudaError_t launch_flash_attn_v2_static_b1_sq4096_sk4032_h16_d128(
    Fav2StaticKernelPtrs params, cudaStream_t stream = 0) {
  using Config = Fav2StaticB1Sq4096Sk4032H16D128Config;
  using LaunchConfig = FlashAttnV2D128LaunchConfig;
  auto kernel_fptr = flash_attn_v2_static_d128_sonny<Config>;

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

inline cudaError_t launch_flash_attn_v2_static_b1_sq4096_sk4096_h16_d128(
    Fav2StaticKernelPtrs params, cudaStream_t stream = 0) {
  using Config = Fav2ProjectB1Sq4096Sk4096H16D128Config;
  using LaunchConfig = FlashAttnV2D128LaunchConfig;
  auto kernel_fptr = flash_attn_v2_static_d128_sonny<Config>;

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

} // namespace cuda_ops_core::detail::sm80_flash_attn_f16_hd128_best
