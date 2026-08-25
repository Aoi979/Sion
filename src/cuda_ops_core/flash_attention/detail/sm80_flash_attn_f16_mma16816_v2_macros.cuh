#pragma once

#define FAV2_SOFTMAX_MAX_REDUCE_4LANES(X)                                  \
  do {                                                                      \
    (X) = fmaxf((X), __shfl_xor_sync(0xffffffffu, (X), 2));               \
    (X) = fmaxf((X), __shfl_xor_sync(0xffffffffu, (X), 1));               \
  } while (0)

#define FAV2_SOFTMAX_SUM_REDUCE_4LANES(X)                                  \
  do {                                                                      \
    (X) += __shfl_xor_sync(0xffffffffu, (X), 2);                           \
    (X) += __shfl_xor_sync(0xffffffffu, (X), 1);                           \
  } while (0)

#ifdef UNFUSE_FMA
#define FAV2_SOFTMAX_EXP2(X, MAX_SCALED)                                   \
  exp2f(__fmul_rn((X), params.softmax_scale_log2) - (MAX_SCALED))
#else
#define FAV2_SOFTMAX_EXP2(X, MAX_SCALED)                                   \
  exp2f((X) * params.softmax_scale_log2 - (MAX_SCALED))
#endif

#define FAV2_SOFTMAX_EXP2_CONVERT_N(N_INDEX)                               \
  do {                                                                      \
    for (int m = 0; m < MMA_M; ++m) {                                       \
      for (int cm = 0; cm < 2; ++cm) {                                     \
        const float max_scaled =                                           \
            row_max[m][cm] == -CUDART_INF_F                                \
                ? 0.0f                                                      \
                : row_max[m][cm] * params.softmax_scale_log2;               \
        for (int cn = 0; cn < 2; ++cn) {                                   \
          for (int c = 0; c < 2; ++c) {                                     \
            tSrS[m][(N_INDEX)][cn][cm][c] =                                 \
                FAV2_SOFTMAX_EXP2(tSrS[m][(N_INDEX)][cn][cm][c],            \
                                  max_scaled);                              \
          }                                                                 \
          const __half2 p =                                                \
              __floats2half2_rn(tSrS[m][(N_INDEX)][cn][cm][0],              \
                                tSrS[m][(N_INDEX)][cn][cm][1]);              \
          as_u32_ref(tOrP[m][(N_INDEX)][cn][cm]) =                          \
              *reinterpret_cast<const uint32_t *>(&p);                      \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  } while (0)

#define FAV2_SOFTMAX_SUM_FIRST_N(N_INDEX)                                  \
  do {                                                                      \
    for (int m = 0; m < MMA_M; ++m) {                                       \
      for (int cm = 0; cm < 2; ++cm) {                                     \
        float sum_value = tSrS[m][(N_INDEX)][0][cm][0];                     \
        sum_value += tSrS[m][(N_INDEX)][0][cm][1];                          \
        sum_value += tSrS[m][(N_INDEX)][1][cm][0];                          \
        sum_value += tSrS[m][(N_INDEX)][1][cm][1];                          \
        row_sum[m][cm] = sum_value;                                         \
      }                                                                     \
    }                                                                       \
  } while (0)

#define FAV2_SOFTMAX_SUM_N(N_INDEX)                                        \
  do {                                                                      \
    for (int m = 0; m < MMA_M; ++m) {                                       \
      for (int cm = 0; cm < 2; ++cm) {                                     \
        float sum_value = row_sum[m][cm];                                  \
        sum_value += tSrS[m][(N_INDEX)][0][cm][0];                          \
        sum_value += tSrS[m][(N_INDEX)][0][cm][1];                          \
        sum_value += tSrS[m][(N_INDEX)][1][cm][0];                          \
        sum_value += tSrS[m][(N_INDEX)][1][cm][1];                          \
        row_sum[m][cm] = sum_value;                                         \
      }                                                                     \
    }                                                                       \
  } while (0)

#define FAV2_LOAD_Q_FRAGMENT(M, STAGE, K)                                 \
  do {                                                                      \
    ldsm::x4<ldsm::N>(                                                       \
        tSrQ[(M)][(STAGE)][0][0], tSrQ[(M)][(STAGE)][0][1],                 \
        tSrQ[(M)][(STAGE)][1][0], tSrQ[(M)][(STAGE)][1][1],                 \
        &sQ[((M) * (kBlockM / MMA_M) + warp_id * 16 + tSsQ_row) *          \
                kSmemStride +                                                \
            tSsQ_col * 8 + (K) * 16]);                                      \
  } while (0)

#define FAV2_LOAD_K_FRAGMENT(N_INDEX, STAGE, K)                            \
  do {                                                                      \
    ldsm::x4<ldsm::N>(                                                       \
        tSrK[(N_INDEX)][(STAGE)][0][0], tSrK[(N_INDEX)][(STAGE)][0][1],     \
        tSrK[(N_INDEX)][(STAGE)][1][0], tSrK[(N_INDEX)][(STAGE)][1][1],     \
        &sK[((N_INDEX) * (kBlockN / MMA_N) + tSsK_row) * kSmemStride +     \
            tSsK_col * 8 + (K) * 16]);                                      \
  } while (0)

#define FAV2_SCORE_MMA_PAIR(M_INDEX, N_INDEX, STAGE)                       \
  do {                                                                      \
    mma::m16n8k16_f32f16f16f32_accum(                                      \
        tSrS[(M_INDEX)][(N_INDEX)][0][0][0],                               \
        tSrS[(M_INDEX)][(N_INDEX)][0][0][1],                               \
        tSrS[(M_INDEX)][(N_INDEX)][0][1][0],                               \
        tSrS[(M_INDEX)][(N_INDEX)][0][1][1],                               \
        tSrQ[(M_INDEX)][(STAGE)][0][0], tSrQ[(M_INDEX)][(STAGE)][0][1],     \
        tSrQ[(M_INDEX)][(STAGE)][1][0], tSrQ[(M_INDEX)][(STAGE)][1][1],     \
        tSrK[(N_INDEX)][(STAGE)][0][0], tSrK[(N_INDEX)][(STAGE)][0][1]);    \
    mma::m16n8k16_f32f16f16f32_accum(                                      \
        tSrS[(M_INDEX)][(N_INDEX)][1][0][0],                               \
        tSrS[(M_INDEX)][(N_INDEX)][1][0][1],                               \
        tSrS[(M_INDEX)][(N_INDEX)][1][1][0],                               \
        tSrS[(M_INDEX)][(N_INDEX)][1][1][1],                               \
        tSrQ[(M_INDEX)][(STAGE)][0][0], tSrQ[(M_INDEX)][(STAGE)][0][1],     \
        tSrQ[(M_INDEX)][(STAGE)][1][0], tSrQ[(M_INDEX)][(STAGE)][1][1],     \
        tSrK[(N_INDEX)][(STAGE)][1][0], tSrK[(N_INDEX)][(STAGE)][1][1]);    \
  } while (0)

#define FAV2_SCORE_MMA_N4(STAGE)                                           \
  do {                                                                      \
    FAV2_SCORE_MMA_PAIR(0, 0, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 1, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 2, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 3, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 0, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 1, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 2, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 3, STAGE);                                      \
  } while (0)

#define FAV2_SCORE_MMA_N8(STAGE)                                           \
  do {                                                                      \
    FAV2_SCORE_MMA_PAIR(0, 0, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 1, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 2, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 3, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 4, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 5, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 6, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(0, 7, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 0, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 1, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 2, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 3, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 4, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 5, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 6, STAGE);                                      \
    FAV2_SCORE_MMA_PAIR(1, 7, STAGE);                                      \
  } while (0)

#define FAV2_LOAD_SCORE_K_N4(K_INDEX, STAGE)                               \
  do {                                                                      \
    FAV2_LOAD_Q_FRAGMENT(0, STAGE, K_INDEX);                               \
    FAV2_LOAD_Q_FRAGMENT(1, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(0, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(1, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(2, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(3, STAGE, K_INDEX);                               \
  } while (0)

#define FAV2_LOAD_SCORE_K_N8(K_INDEX, STAGE)                               \
  do {                                                                      \
    FAV2_LOAD_SCORE_K_N4(K_INDEX, STAGE);                                  \
    FAV2_LOAD_K_FRAGMENT(4, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(5, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(6, STAGE, K_INDEX);                               \
    FAV2_LOAD_K_FRAGMENT(7, STAGE, K_INDEX);                               \
  } while (0)

#define FAV2_LOAD_V_FRAGMENT(K_INDEX, N_INDEX, STAGE)                      \
  do {                                                                      \
    ldsm::x4<ldsm::T>(                                                       \
        tOrV[(K_INDEX)][(STAGE)][0][0], tOrV[(K_INDEX)][(STAGE)][0][1],    \
        tOrV[(K_INDEX)][(STAGE)][1][0], tOrV[(K_INDEX)][(STAGE)][1][1],    \
        &sV[(K_INDEX) * (kHeadDim / MMA_K) + tOsV_col +                    \
            (tOsV_row + (N_INDEX) * 16) * kSmemStride]);                   \
  } while (0)

#define FAV2_LOAD_V_STAGE_4(N_INDEX, STAGE)                                \
  do {                                                                      \
    FAV2_LOAD_V_FRAGMENT(0, N_INDEX, STAGE);                               \
    FAV2_LOAD_V_FRAGMENT(1, N_INDEX, STAGE);                               \
    FAV2_LOAD_V_FRAGMENT(2, N_INDEX, STAGE);                               \
    FAV2_LOAD_V_FRAGMENT(3, N_INDEX, STAGE);                               \
  } while (0)

#define FAV2_LOAD_V_STAGE_8(N_INDEX, STAGE)                                \
  do {                                                                      \
    FAV2_LOAD_V_STAGE_4(N_INDEX, STAGE);                                   \
    FAV2_LOAD_V_FRAGMENT(4, N_INDEX, STAGE);                               \
    FAV2_LOAD_V_FRAGMENT(5, N_INDEX, STAGE);                               \
    FAV2_LOAD_V_FRAGMENT(6, N_INDEX, STAGE);                               \
    FAV2_LOAD_V_FRAGMENT(7, N_INDEX, STAGE);                               \
  } while (0)

#define FAV2_OUTPUT_MMA_PAIR(M_INDEX, K_INDEX, N_INDEX, STAGE)             \
  do {                                                                      \
    mma::m16n8k16_f32f16f16f32_accum(                                      \
        tOrO[(M_INDEX)][(K_INDEX)][0][0][0],                               \
        tOrO[(M_INDEX)][(K_INDEX)][0][0][1],                               \
        tOrO[(M_INDEX)][(K_INDEX)][0][1][0],                               \
        tOrO[(M_INDEX)][(K_INDEX)][0][1][1],                               \
        tOrP[(M_INDEX)][(N_INDEX)][0][0],                                  \
        tOrP[(M_INDEX)][(N_INDEX)][0][1],                                  \
        tOrP[(M_INDEX)][(N_INDEX)][1][0],                                  \
        tOrP[(M_INDEX)][(N_INDEX)][1][1],                                  \
        tOrV[(K_INDEX)][(STAGE)][0][0],                                    \
        tOrV[(K_INDEX)][(STAGE)][0][1]);                                   \
    mma::m16n8k16_f32f16f16f32_accum(                                      \
        tOrO[(M_INDEX)][(K_INDEX)][1][0][0],                               \
        tOrO[(M_INDEX)][(K_INDEX)][1][0][1],                               \
        tOrO[(M_INDEX)][(K_INDEX)][1][1][0],                               \
        tOrO[(M_INDEX)][(K_INDEX)][1][1][1],                               \
        tOrP[(M_INDEX)][(N_INDEX)][0][0],                                  \
        tOrP[(M_INDEX)][(N_INDEX)][0][1],                                  \
        tOrP[(M_INDEX)][(N_INDEX)][1][0],                                  \
        tOrP[(M_INDEX)][(N_INDEX)][1][1],                                  \
        tOrV[(K_INDEX)][(STAGE)][1][0],                                    \
        tOrV[(K_INDEX)][(STAGE)][1][1]);                                   \
  } while (0)

#define FAV2_OUTPUT_STAGE_4(N_INDEX, STAGE)                                \
  do {                                                                      \
    FAV2_OUTPUT_MMA_PAIR(0, 0, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(0, 1, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(0, 2, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(0, 3, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 0, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 1, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 2, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 3, N_INDEX, STAGE);                            \
  } while (0)

#define FAV2_OUTPUT_STAGE_8(N_INDEX, STAGE)                                \
  do {                                                                      \
    FAV2_OUTPUT_STAGE_4(N_INDEX, STAGE);                                   \
    FAV2_OUTPUT_MMA_PAIR(0, 4, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(0, 5, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(0, 6, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(0, 7, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 4, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 5, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 6, N_INDEX, STAGE);                            \
    FAV2_OUTPUT_MMA_PAIR(1, 7, N_INDEX, STAGE);                            \
  } while (0)
