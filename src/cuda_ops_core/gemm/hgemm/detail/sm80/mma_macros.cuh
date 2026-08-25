#pragma once

#define MMA_1_ROW_SLOT(m, k_slot)                                              \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][0][0][0][0]), as_u32(tCrC[m][0][0][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[0][(k_slot)][0][0][0]),                                      \
      as_u32(tCrB[0][(k_slot)][0][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][0][1][0][0]), as_u32(tCrC[m][0][1][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[0][(k_slot)][1][0][0]),                                      \
      as_u32(tCrB[0][(k_slot)][1][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][1][0][0][0]), as_u32(tCrC[m][1][0][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[1][(k_slot)][0][0][0]),                                      \
      as_u32(tCrB[1][(k_slot)][0][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][1][1][0][0]), as_u32(tCrC[m][1][1][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[1][(k_slot)][1][0][0]),                                      \
      as_u32(tCrB[1][(k_slot)][1][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][2][0][0][0]), as_u32(tCrC[m][2][0][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[2][(k_slot)][0][0][0]),                                      \
      as_u32(tCrB[2][(k_slot)][0][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][2][1][0][0]), as_u32(tCrC[m][2][1][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[2][(k_slot)][1][0][0]),                                      \
      as_u32(tCrB[2][(k_slot)][1][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][3][0][0][0]), as_u32(tCrC[m][3][0][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[3][(k_slot)][0][0][0]),                                      \
      as_u32(tCrB[3][(k_slot)][0][1][0]));                                     \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][3][1][0][0]), as_u32(tCrC[m][3][1][1][0]),                \
      as_u32(tCrA[m][(k_slot)][0][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][0][1][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][0][0]),                                      \
      as_u32(tCrA[m][(k_slot)][1][1][0]),                                      \
      as_u32(tCrB[3][(k_slot)][1][0][0]),                                      \
      as_u32(tCrB[3][(k_slot)][1][1][0]))


#define MMA_1_ROW_SLOT_FP32(m, k_slot)                              \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][0][0][0], tCrC[m][0][0][0][1], tCrC[m][0][0][1][0],           \
      tCrC[m][0][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][0][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][0][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][1][0][0], tCrC[m][0][1][0][1], tCrC[m][0][1][1][0],           \
      tCrC[m][0][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][0][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][0][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][0][0][0], tCrC[m][1][0][0][1], tCrC[m][1][0][1][0],           \
      tCrC[m][1][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][1][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][1][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][1][0][0], tCrC[m][1][1][0][1], tCrC[m][1][1][1][0],           \
      tCrC[m][1][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][1][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][1][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][0][0][0], tCrC[m][2][0][0][1], tCrC[m][2][0][1][0],           \
      tCrC[m][2][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][2][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][2][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][1][0][0], tCrC[m][2][1][0][1], tCrC[m][2][1][1][0],           \
      tCrC[m][2][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][2][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][2][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][0][0][0], tCrC[m][3][0][0][1], tCrC[m][3][0][1][0],           \
      tCrC[m][3][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][3][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][3][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][1][0][0], tCrC[m][3][1][0][1], tCrC[m][3][1][1][0],           \
      tCrC[m][3][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][3][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][3][1][1][0]))

// Optional reverse-N traversal. It may expose more generated .reuse markers,
// but that does not guarantee a performance benefit; the default path does
// not enable this variant.
#define MMA_1_ROW_SLOT_FP32_REV(m, k_slot)                              \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][1][0][0], tCrC[m][3][1][0][1], tCrC[m][3][1][1][0],           \
      tCrC[m][3][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][3][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][3][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][0][0][0], tCrC[m][3][0][0][1], tCrC[m][3][0][1][0],           \
      tCrC[m][3][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][3][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][3][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][1][0][0], tCrC[m][2][1][0][1], tCrC[m][2][1][1][0],           \
      tCrC[m][2][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][2][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][2][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][0][0][0], tCrC[m][2][0][0][1], tCrC[m][2][0][1][0],           \
      tCrC[m][2][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][2][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][2][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][1][0][0], tCrC[m][1][1][0][1], tCrC[m][1][1][1][0],           \
      tCrC[m][1][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][1][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][1][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][0][0][0], tCrC[m][1][0][0][1], tCrC[m][1][0][1][0],           \
      tCrC[m][1][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][1][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][1][0][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][1][0][0], tCrC[m][0][1][0][1], tCrC[m][0][1][1][0],           \
      tCrC[m][0][1][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][0][1][0][0]),                                      \
      as_u32(tCrB[(k_slot)][0][1][1][0]));                                     \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][0][0][0], tCrC[m][0][0][0][1], tCrC[m][0][0][1][0],           \
      tCrC[m][0][0][1][1], as_u32(tCrA[(k_slot)][m][0][0][0]),                 \
      as_u32(tCrA[(k_slot)][m][0][1][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][0][0]),                                      \
      as_u32(tCrA[(k_slot)][m][1][1][0]),                                      \
      as_u32(tCrB[(k_slot)][0][0][0][0]),                                      \
      as_u32(tCrB[(k_slot)][0][0][1][0]))
