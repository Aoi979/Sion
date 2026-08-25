#pragma once
#include "../detail/sm80_mma_traits.cuh"
#include "../detail/sm80_flash_attn_f16_mma16816_v2_macros.cuh"
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/swizzle.hpp>
#include <math_constants.h>

namespace sm80_flash_attn_v2 {

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

template <int kHeadDim, int kSmemStride, int kBlockX, int kThreads>
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

    cp_async::cg<16>(&sX[row * kSmemStride + col],
                     &gX[row * x_row_stride + col]);
  }

  cp_async::commit_group();
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

__device__ __forceinline__ uint32_t fav2_pack_f32x2_to_f16x2(float x, float y) {
  const __half2 xy = __floats2half2_rn(x, y);
  return *reinterpret_cast<const uint32_t *>(&xy);
}

template <int kHeadDim, int kSmemStride, int kBlockM, int kWarps, int kThreads,
          int MMA_M,
          int MMA_K>
__device__ __forceinline__ void
store_output_epilogue(half *sO, half *gO, int o_row_stride,
                      const float (&tOrO)[MMA_M][MMA_K][2][2][2]) {
  constexpr int kElementsPerStore = 8;
  constexpr int kSmemStrideO = kSmemStride;
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
__global__ void flash_attn_v2(FlashFwdParams<kHeadDim> params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;

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
  static_assert(kHeadDim == 64 || kHeadDim == 128);
  static_assert(MMA_M == 2);
  static_assert((kHeadDim == 64 && MMA_N == 8 && MMA_K == 4) ||
                (kHeadDim == 128 && MMA_N == 4 && MMA_K == 8));
  constexpr int kSmemPad = kHeadDim == 128 ? 8 : 0;
  constexpr int kSmemStride = kHeadDim + kSmemPad;

  // Launch/params preconditions:
  // - blockDim.x == kThreads.
  // - grid = (params.seqlen_q / kBlockM, params.batch_size, params.heads_q).
  // - params.seqlen_q % kBlockM == 0 and params.seqlen_k % kBlockN == 0.
  // - params.heads_q == params.heads_k * params.q_heads_per_kv_head.
  // - Q/K/V/O base pointers are non-null and 16B-aligned.
  // - Q/K/V/O row strides are positive and multiples of 8 half elements.

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
  half *sK = sQ + kBlockM * kSmemStride;
  half *sV = sK + kBlockN * kSmemStride;

  async_load_x_tensor<kHeadDim, kSmemStride, kBlockM, kThreads>(
      sQ, gQ, QueryRowStride);
  async_load_x_tensor<kHeadDim, kSmemStride, kBlockN, kThreads>(
      sK, gK_base, KeyRowStride);

  float tOrO[MMA_M][MMA_K][2][2][2];
  clearO(tOrO);

  const int lane_id = threadIdx.x % 32;
  const int tSsQ_row = lane_id % 16;
  const int tSsQ_col = lane_id / 16;
  const int tSsK_row = (lane_id / 16) * 8 + lane_id % 8;
  const int tSsK_col = (lane_id % 16) / 8;
  const int warp_id = threadIdx.x / 32;
  const int tOsV_row = lane_id % 16;
  const int tOsV_col = (lane_id / 16) * 8;

  // Wait Q and K_0 ready.
  cp_async::wait_group<0>();
  __syncthreads();
  const int num_k_tiles = params.seqlen_k / kBlockN;
  {
    float row_max[MMA_M][2];
    float row_sum[MMA_M][2];

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      const half *gV = gV_base + k_tile * kBlockN * ValueRowStride;

      // Load V_current.
      async_load_x_tensor<kHeadDim, kSmemStride, kBlockN, kThreads>(
          sV, gV, ValueRowStride);

      float tSrS[MMA_M][MMA_N][2][2][2];

      {
        // (MMA_M, Stage=2, CoreK, CoreM, Core)
        half tSrQ[MMA_M][2][2][2][2];
        // (MMA_N, Stage=2, CoreN, CoreK, Core)
        half tSrK[MMA_N][2][2][2][2];

        // acc_s = Q_i @ K_current^T
        clearS(tSrS);

        if constexpr (kHeadDim == 64) {
          FAV2_LOAD_SCORE_K_N8(0, 0);
          FAV2_LOAD_SCORE_K_N8(1, 1);
          FAV2_SCORE_MMA_N8(0);
          FAV2_LOAD_SCORE_K_N8(2, 0);
          FAV2_SCORE_MMA_N8(1);
          FAV2_LOAD_SCORE_K_N8(3, 1);
          FAV2_SCORE_MMA_N8(0);
          FAV2_SCORE_MMA_N8(1);
        } else {
          FAV2_LOAD_SCORE_K_N4(0, 0);
          FAV2_LOAD_SCORE_K_N4(1, 1);
          FAV2_SCORE_MMA_N4(0);
          FAV2_LOAD_SCORE_K_N4(2, 0);
          FAV2_SCORE_MMA_N4(1);
          FAV2_LOAD_SCORE_K_N4(3, 1);
          FAV2_SCORE_MMA_N4(0);
          FAV2_LOAD_SCORE_K_N4(4, 0);
          FAV2_SCORE_MMA_N4(1);
          FAV2_LOAD_SCORE_K_N4(5, 1);
          FAV2_SCORE_MMA_N4(0);
          FAV2_LOAD_SCORE_K_N4(6, 0);
          FAV2_SCORE_MMA_N4(1);
          FAV2_LOAD_SCORE_K_N4(7, 1);
          FAV2_SCORE_MMA_N4(0);
          FAV2_SCORE_MMA_N4(1);
        }
      }

#pragma unroll
      for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
          float max_value = tSrS[m][0][0][cm][0];
#pragma unroll
          for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
            for (int cn = 0; cn < 2; ++cn) {
#pragma unroll
              for (int c = 0; c < 2; ++c) {
                if (n != 0 || cn != 0 || c != 0) {
                  max_value = fmaxf(max_value, tSrS[m][n][cn][cm][c]);
                }
              }
            }
          }
          FAV2_SOFTMAX_MAX_REDUCE_4LANES(max_value);

          if (k_tile != 0) {
            const float max_prev = row_max[m][cm];
            max_value = fmaxf(max_prev, max_value);
            const float scores_scale =
                exp2f((max_prev - max_value) * params.softmax_scale_log2);
            row_sum[m][cm] *= scores_scale;

#pragma unroll
            for (int k = 0; k < MMA_K; ++k) {
#pragma unroll
              for (int ck = 0; ck < 2; ++ck) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                  tOrO[m][k][ck][cm][c] *= scores_scale;
                }
              }
            }
          }

          row_max[m][cm] = max_value;
        }
      }

      // Need V_current before P @ V.
      cp_async::wait_group<0>();
      __syncthreads();

      // Current K is no longer needed after score computation.
      if (k_tile + 1 < num_k_tiles) {
        const half *gK_next = gK_base + (k_tile + 1) * kBlockN * KeyRowStride;

        async_load_x_tensor<kHeadDim, kSmemStride, kBlockN, kThreads>(
            sK, gK_next, KeyRowStride);
      }

      {
        // (MMA_M, MMA_N, CoreN, CoreM, Core)
        half tOrP[MMA_M][MMA_N][2][2][2];
        // (MMA_K, Stage=2, CoreK, CoreN, Core)
        half tOrV[MMA_K][2][2][2][2];

        if constexpr (kHeadDim == 64) {
          FAV2_SOFTMAX_EXP2_CONVERT_N(0);
          if (k_tile == 0) {
            FAV2_SOFTMAX_SUM_FIRST_N(0);
          } else {
            FAV2_SOFTMAX_SUM_N(0);
          }
          FAV2_LOAD_V_STAGE_4(0, 0);
          FAV2_LOAD_V_STAGE_4(1, 1);
          FAV2_OUTPUT_STAGE_4(0, 0);
          FAV2_SOFTMAX_EXP2_CONVERT_N(1);
          FAV2_SOFTMAX_SUM_N(1);
          FAV2_LOAD_V_STAGE_4(2, 0);
          FAV2_OUTPUT_STAGE_4(1, 1);
          FAV2_SOFTMAX_EXP2_CONVERT_N(2);
          FAV2_SOFTMAX_SUM_N(2);
          FAV2_LOAD_V_STAGE_4(3, 1);
          FAV2_OUTPUT_STAGE_4(2, 0);
          FAV2_SOFTMAX_EXP2_CONVERT_N(3);
          FAV2_SOFTMAX_SUM_N(3);
          FAV2_LOAD_V_STAGE_4(4, 0);
          FAV2_OUTPUT_STAGE_4(3, 1);
          FAV2_SOFTMAX_EXP2_CONVERT_N(4);
          FAV2_SOFTMAX_SUM_N(4);
          FAV2_LOAD_V_STAGE_4(5, 1);
          FAV2_OUTPUT_STAGE_4(4, 0);
          FAV2_SOFTMAX_EXP2_CONVERT_N(5);
          FAV2_SOFTMAX_SUM_N(5);
          FAV2_LOAD_V_STAGE_4(6, 0);
          FAV2_OUTPUT_STAGE_4(5, 1);
          FAV2_SOFTMAX_EXP2_CONVERT_N(6);
          FAV2_SOFTMAX_SUM_N(6);
          FAV2_LOAD_V_STAGE_4(7, 1);
          FAV2_OUTPUT_STAGE_4(6, 0);
          FAV2_SOFTMAX_EXP2_CONVERT_N(7);
          FAV2_SOFTMAX_SUM_N(7);
          FAV2_OUTPUT_STAGE_4(7, 1);
        } else {
          FAV2_SOFTMAX_EXP2_CONVERT_N(0);
          if (k_tile == 0) {
            FAV2_SOFTMAX_SUM_FIRST_N(0);
          } else {
            FAV2_SOFTMAX_SUM_N(0);
          }
          FAV2_LOAD_V_STAGE_8(0, 0);
          FAV2_LOAD_V_STAGE_8(1, 1);
          FAV2_OUTPUT_STAGE_8(0, 0);
          FAV2_SOFTMAX_EXP2_CONVERT_N(1);
          FAV2_SOFTMAX_SUM_N(1);
          FAV2_LOAD_V_STAGE_8(2, 0);
          FAV2_OUTPUT_STAGE_8(1, 1);
          FAV2_SOFTMAX_EXP2_CONVERT_N(2);
          FAV2_SOFTMAX_SUM_N(2);
          FAV2_LOAD_V_STAGE_8(3, 1);
          FAV2_OUTPUT_STAGE_8(2, 0);
          FAV2_SOFTMAX_EXP2_CONVERT_N(3);
          FAV2_SOFTMAX_SUM_N(3);
          FAV2_OUTPUT_STAGE_8(3, 1);
        }
      }

      // Before next iteration, K_next must be ready in sK.
      if (k_tile + 1 < num_k_tiles) {
        cp_async::wait_group<0>();
        __syncthreads();
      }
    }

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int cm = 0; cm < 2; ++cm) {
      FAV2_SOFTMAX_SUM_REDUCE_4LANES(row_sum[m][cm]);
    }
  }

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
            tOrO[m][k][ck][cm][c] *= inv_sum;
          }
        }
      }
    }
  }

  }

  store_output_epilogue<kHeadDim, kSmemStride, kBlockM, kWarps, kThreads,
                        MMA_M, MMA_K>(sQ, gO, OutputRowStride, tOrO);
}

#undef FAV2_SOFTMAX_MAX_REDUCE_4LANES
#undef FAV2_SOFTMAX_SUM_REDUCE_4LANES
#undef FAV2_SOFTMAX_EXP2
#undef FAV2_SOFTMAX_EXP2_CONVERT_N
#undef FAV2_SOFTMAX_SUM_FIRST_N
#undef FAV2_SOFTMAX_SUM_N
#undef FAV2_LOAD_Q_FRAGMENT
#undef FAV2_LOAD_K_FRAGMENT
#undef FAV2_SCORE_MMA_PAIR
#undef FAV2_SCORE_MMA_N4
#undef FAV2_SCORE_MMA_N8
#undef FAV2_LOAD_SCORE_K_N4
#undef FAV2_LOAD_SCORE_K_N8
#undef FAV2_LOAD_V_FRAGMENT
#undef FAV2_LOAD_V_STAGE_4
#undef FAV2_LOAD_V_STAGE_8
#undef FAV2_OUTPUT_MMA_PAIR
#undef FAV2_OUTPUT_STAGE_4
#undef FAV2_OUTPUT_STAGE_8

namespace fav2_sm80 {

template <int kHeadDim, int kBlockM, int kBlockN>
struct FlashAttnV2LaunchConfig {
  static constexpr int kHeadDimValue = kHeadDim;
  static constexpr int kBlockMValue = kBlockM;
  static constexpr int kBlockNValue = kBlockN;
  static constexpr int kWarps = 4;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kSmemPad = kHeadDim == 128 ? 8 : 0;
  static constexpr int kSmemStride = kHeadDim + kSmemPad;
  static constexpr int kSmemBytes =
      (kBlockM + 2 * kBlockN) * kSmemStride * sizeof(half);
};

template <int kHeadDim> struct FlashAttnV2Sm80Config {
  static constexpr bool kSupported = false;
  static constexpr int kBlockMValue = 0;
  static constexpr int kBlockNValue = 0;
};

// Wired subset of the official FlashAttention SM80 non-dropout configs.
// The official hdim96 config also uses 128x64x4, but this loader currently
// requires kThreads % (kHeadDim / 8) == 0. Official hdim192/256 use 8 warps on
// large SM80 GPUs, while this handwritten kernel is fixed at 4 warps.
template <>
struct FlashAttnV2Sm80Config<64> : FlashAttnV2LaunchConfig<64, 128, 128> {
  static constexpr bool kSupported = true;
};

template <>
struct FlashAttnV2Sm80Config<128> : FlashAttnV2LaunchConfig<128, 128, 64> {
  static constexpr bool kSupported = true;
};

template <int kHeadDim, int kBlockM, int kBlockN>
inline cudaError_t launch_flash_attn_v2_config(FlashFwdParams<kHeadDim> params,
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
  static_assert((kBlockM * kHeadDim / kElementsPerAccess) % Config::kThreads ==
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
inline cudaError_t launch_flash_attn_v2_sm80(FlashFwdParams<kHeadDim> params,
                                             cudaStream_t stream = 0) {
  using Config = FlashAttnV2Sm80Config<kHeadDim>;
  static_assert(
      Config::kSupported,
      "SM80 launch config is wired for head_dim 64 and 128 in this "
      "handwritten kernel.");

  return launch_flash_attn_v2_config<kHeadDim, Config::kBlockMValue,
                                     Config::kBlockNValue>(params, stream);
}

} // namespace fav2_sm80

} // namespace sm80_flash_attn_v2
