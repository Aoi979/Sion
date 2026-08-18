#pragma once

/* SM80 128x256 Stream-K FP32-accumulate HGEMM device implementation.
 *
 * The CudaOpsCore launcher owns validation, workspace lifetime, and registry
 * integration; this header contains the scheduler and CUDA kernel logic.
 */


#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuda_ops_core::detail::sm80_hgemm_128x256_streamk {

__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

#define MMA_1_ROW(m)                                                           \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][0][0][0][0]), as_u32(tCrC[m][0][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[0][k_block][0][0][0]), as_u32(tCrB[0][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][0][1][0][0]), as_u32(tCrC[m][0][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[0][k_block][1][0][0]), as_u32(tCrB[0][k_block][1][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][1][0][0][0]), as_u32(tCrC[m][1][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[1][k_block][0][0][0]), as_u32(tCrB[1][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][1][1][0][0]), as_u32(tCrC[m][1][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[1][k_block][1][0][0]), as_u32(tCrB[1][k_block][1][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][2][0][0][0]), as_u32(tCrC[m][2][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[2][k_block][0][0][0]), as_u32(tCrB[2][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][2][1][0][0]), as_u32(tCrC[m][2][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[2][k_block][1][0][0]), as_u32(tCrB[2][k_block][1][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][3][0][0][0]), as_u32(tCrC[m][3][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[3][k_block][0][0][0]), as_u32(tCrB[3][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][3][1][0][0]), as_u32(tCrC[m][3][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[3][k_block][1][0][0]), as_u32(tCrB[3][k_block][1][1][0]))

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

#define MMA_1_ROW_FP32(m)                                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][0][0][0], tCrC[m][0][0][0][1], tCrC[m][0][0][1][0],           \
      tCrC[m][0][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[0][k_block][0][0][0]),    \
      as_u32(tCrB[0][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][1][0][0], tCrC[m][0][1][0][1], tCrC[m][0][1][1][0],           \
      tCrC[m][0][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[0][k_block][1][0][0]),    \
      as_u32(tCrB[0][k_block][1][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][0][0][0], tCrC[m][1][0][0][1], tCrC[m][1][0][1][0],           \
      tCrC[m][1][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[1][k_block][0][0][0]),    \
      as_u32(tCrB[1][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][1][0][0], tCrC[m][1][1][0][1], tCrC[m][1][1][1][0],           \
      tCrC[m][1][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[1][k_block][1][0][0]),    \
      as_u32(tCrB[1][k_block][1][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][0][0][0], tCrC[m][2][0][0][1], tCrC[m][2][0][1][0],           \
      tCrC[m][2][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[2][k_block][0][0][0]),    \
      as_u32(tCrB[2][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][1][0][0], tCrC[m][2][1][0][1], tCrC[m][2][1][1][0],           \
      tCrC[m][2][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[2][k_block][1][0][0]),    \
      as_u32(tCrB[2][k_block][1][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][0][0][0], tCrC[m][3][0][0][1], tCrC[m][3][0][1][0],           \
      tCrC[m][3][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[3][k_block][0][0][0]),    \
      as_u32(tCrB[3][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][1][0][0], tCrC[m][3][1][0][1], tCrC[m][3][1][1][0],           \
      tCrC[m][3][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[3][k_block][1][0][0]),    \
      as_u32(tCrB[3][k_block][1][1][0]))

#define MMA_1_ROW_SLOT_FP32_SWAPPED(m, k_slot)                                         \
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

// Experimental variant: traverse the B fragments in reverse order.  Pairing
// the normal and reverse traversals across adjacent A fragments makes the
// last B fragment of one MMA group the first B fragment of the next group.
// This is intended to give ptxas an opportunity to emit HMMA B-operand
// register-reuse flags, matching the serpentine pattern seen in cuBLAS SASS.
#define MMA_1_ROW_SLOT_FP32_SWAPPED_REV(m, k_slot)                              \
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

namespace cp_async {

enum class CacheMode {
  CA, // cache all: L1 + L2
  CG  // cache global: L2 only
};

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

template <int N> __device__ __forceinline__ void wait_group() {
  static_assert(N >= 0 && N <= 7, "cp.async.wait_group N must be in [0, 7]");
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <CacheMode Mode, int Bytes>
__device__ __forceinline__ void copy(void *smem_ptr, const void *gmem_ptr) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                "cp.async.ca supports 4, 8, 16 bytes; cp.async.cg supports "
                "only 16 bytes");

  if constexpr (Mode == CacheMode::CA) {
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
                 :
                 : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr), "n"(Bytes));
  } else {
    static_assert(Bytes == 16, "cp.async.cg only supports 16 bytes");

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr));
  }
}

template <int Bytes>
__device__ __forceinline__ void ca(void *smem_ptr, const void *gmem_ptr) {
  copy<CacheMode::CA, Bytes>(smem_ptr, gmem_ptr);
}

template <int Bytes>
__device__ __forceinline__ void cg(void *smem_ptr, const void *gmem_ptr) {
  copy<CacheMode::CG, Bytes>(smem_ptr, gmem_ptr);
}

} // namespace cp_async

namespace ldsm {

enum class Trans { No, Yes };

constexpr Trans T = Trans::Yes;
constexpr Trans N = Trans::No;

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x1(uint32_t &d0, const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
                 "{%0}, [%1];\n"
                 : "=r"(d0)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
                 "{%0}, [%1];\n"
                 : "=r"(d0)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x2(uint32_t &d0, uint32_t &d1,
                                   const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x4(uint32_t &v0v1, uint32_t &v2v3,
                                   uint32_t &v4v5, uint32_t &v6v7,
                                   const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0v1), "=r"(v2v3), "=r"(v4v5), "=r"(v6v7)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0v1), "=r"(v2v3), "=r"(v4v5), "=r"(v6v7)
                 : "r"(addr));
  }
}

} // namespace ldsm

namespace mma {

__device__ __forceinline__ void
m16n8k16_f16f16f16(uint32_t &d0, uint32_t &d1,

                   uint32_t const &a0, uint32_t const &a1, uint32_t const &a2,
                   uint32_t const &a3,

                   uint32_t const &b0, uint32_t const &b1,

                   uint32_t const &c0, uint32_t const &c1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "
               "{%2, %3, %4, %5}, "
               "{%6, %7}, "
               "{%8, %9};\n"
               : "=r"(d0), "=r"(d1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0),
                 "r"(c1));
}

__device__ __forceinline__ void
m16n8k16_f16f16f16_accum(uint32_t &c0, uint32_t &c1,

                         uint32_t const &a0, uint32_t const &a1,
                         uint32_t const &a2, uint32_t const &a3,

                         uint32_t const &b0, uint32_t const &b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "
               "{%2, %3, %4, %5}, "
               "{%6, %7}, "
               "{%0, %1};\n"
               : "+r"(c0), "+r"(c1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void
m16n8k16_f32f16f16f32_accum(float &c0, float &c1, float &c2, float &c3,

                            uint32_t const &a0, uint32_t const &a1,
                            uint32_t const &a2, uint32_t const &a3,

                            uint32_t const &b0, uint32_t const &b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

} // namespace mma

__device__ __forceinline__ uint32_t &as_u32(half &x) {
  return *reinterpret_cast<uint32_t *>(&x);
}

__device__ __forceinline__ uint32_t pack_f32x2_to_f16x2(float x, float y) {
  __half2 xy = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t *>(&xy);
}

namespace hgemm_smem {

__device__ __forceinline__ int offset_A(int m, int k) {
  int k_vec = (k >> 3) ^ (m & 7);
  return (m << 6) + (k_vec << 3) + (k & 7);
}

__device__ __forceinline__ int offset_B(int n, int k) {
  int n_vec = (n >> 3) ^ (k & 7);
  return (k << 7) + (n_vec << 3) + (n & 7);
}

} // namespace hgemm_smem

namespace hgemm_epilogue {

template <int StoreIter, int kCtaN, int kElementsPerAccess, int kThreads,
          int kSmemStrideC>
__device__ __forceinline__ void store_gmem_vec(half *gC, const half *sC,
                                               int strideC) {
  constexpr int kVecsPerRow = kCtaN / kElementsPerAccess;
  int vec = threadIdx.x + StoreIter * kThreads;
  int vec_row = vec / kVecsPerRow;
  int vec_col = vec % kVecsPerRow;
  uint4 *d_ptr = reinterpret_cast<uint4 *>(gC + vec_row * strideC +
                                           vec_col * kElementsPerAccess);
  const uint4 *s_ptr =
      reinterpret_cast<const uint4 *>(sC + vec_row * kSmemStrideC +
                                      vec_col * kElementsPerAccess);
  *d_ptr = *s_ptr;
}

template <int StoreIter, int kStoreIterations, int kCtaN,
          int kElementsPerAccess, int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_unrolled(half *gC, const half *sC,
                                                    int strideC) {
  store_gmem_vec<StoreIter, kCtaN, kElementsPerAccess, kThreads, kSmemStrideC>(
      gC, sC, strideC);
  if constexpr (StoreIter + 1 < kStoreIterations) {
    store_gmem_unrolled<StoreIter + 1, kStoreIterations, kCtaN,
                        kElementsPerAccess, kThreads, kSmemStrideC>(
        gC, sC, strideC);
  }
}

// Experimental writeback path: expose the invariant per-thread base address
// and advance both pointers by one fixed tile-row step.  The global step is
// runtime-valued because strideC is the problem N, so it cannot be encoded as
// an STG immediate offset.  The loop may be fully unrolled: the important
// difference from store_gmem_unrolled is that the unrolled instances carry a
// pointer recurrence instead of rebuilding row*strideC independently.
template <int kStoreIterations, int kCtaN, int kElementsPerAccess,
          int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_recurrent(half *gC, const half *sC,
                                                     int strideC) {
  constexpr int kVecsPerRow = kCtaN / kElementsPerAccess;
  constexpr int kRowsPerStep = kThreads / kVecsPerRow;

  int vec_row = threadIdx.x / kVecsPerRow;
  int vec_col = threadIdx.x % kVecsPerRow;
  half *d_ptr = gC + vec_row * strideC + vec_col * kElementsPerAccess;
  const half *s_ptr = sC + vec_row * kSmemStrideC +
                      vec_col * kElementsPerAccess;
  int d_step = kRowsPerStep * strideC;
  constexpr int s_step = kRowsPerStep * kSmemStrideC;

#pragma unroll
  for (int i = 0; i < kStoreIterations; ++i) {
    *reinterpret_cast<uint4 *>(d_ptr) =
        *reinterpret_cast<const uint4 *>(s_ptr);
    d_ptr += d_step;
    s_ptr += s_step;
  }
}

} // namespace hgemm_epilogue

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_A(half *smem_A, const half *gA,
                                                 int tA_row, int tA_col,
                                                 int strideA) {
  constexpr int kElementsPerAccess = 8;
  int row = tA_row + RowBlock * 16;
  int col = tA_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_A[hgemm_smem::offset_A(row, col)],
                   &gA[row * strideA + col]);
}

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_B(half *smem_B, const half *gB,
                                                 int tB_row, int tB_col,
                                                 int strideB) {
  constexpr int kElementsPerAccess = 8;
  int row = tB_row + RowBlock * 8;
  int col = tB_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_B[hgemm_smem::offset_B(col, row)],
                   &gB[row * strideB + col]);
}

template <typename Shape_MNK> struct Buffer {
  half A[Shape_MNK::M * Shape_MNK::K];
  half B[Shape_MNK::K * Shape_MNK::N];
};
template <typename Shape_MNK, int Stages> struct HgemmSharedStorage {
  Buffer<Shape_MNK> buffer[Stages];
};

struct shape_mnk {
  static constexpr int M = 128;
  static constexpr int N = 128;
  static constexpr int K = 64;
};

struct shape_mnk_n256 {
  static constexpr int M = 128;
  static constexpr int N = 256;
  static constexpr int K = 64;
};

// only supports M/N/K that are multiples of the CTA shape.


namespace n256_splitk {

__device__ __forceinline__ int offset_B(int n, int k) {
  int n_vec = (n >> 3) ^ (k & 7);
  return (k << 8) + (n_vec << 3) + (n & 7);
}

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_A(half *smem_A, const half *gA,
                                                 int tA_row, int tA_col,
                                                 int strideA) {
  constexpr int kElementsPerAccess = 8;
  int row = tA_row + RowBlock * 32;
  int col = tA_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_A[hgemm_smem::offset_A(row, col)],
                   &gA[row * strideA + col]);
}

// The four A row blocks differ by 32 rows.  Since 32 is a multiple of the
// swizzle period (8), offset_A shares the same swizzled column term for all
// four rows; only the row-major base advances by 32 * 64 half elements.
__device__ __forceinline__ void issue_cp_async_A4(half *smem_A, const half *gA,
                                                  int tA_row, int tA_col,
                                                  int strideA) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kSmemRowStep = 32 * 64;
  int col = tA_col * kElementsPerAccess;
  int smem_base = hgemm_smem::offset_A(tA_row, col);
  const char *gA_bytes = reinterpret_cast<const char *>(gA);
  int gmem_base_bytes = (tA_row * strideA + col) * sizeof(half);
  int gmem_row_step_bytes = 32 * strideA * sizeof(half);
  const char *gA0 = gA_bytes + gmem_base_bytes;
  const char *gA1 = gA0 + gmem_row_step_bytes;
  const char *gA2 = gA1 + gmem_row_step_bytes;
  const char *gA3 = gA2 + gmem_row_step_bytes;

  cp_async::cg<16>(&smem_A[smem_base + 0 * kSmemRowStep],
                   gA0);
  cp_async::cg<16>(&smem_A[smem_base + 1 * kSmemRowStep],
                   gA1);
  cp_async::cg<16>(&smem_A[smem_base + 2 * kSmemRowStep],
                   gA2);
  cp_async::cg<16>(&smem_A[smem_base + 3 * kSmemRowStep],
                   gA3);
}

// The eight B row blocks differ by 8 rows.  For this layout, row + 8*i
// keeps the swizzled column term unchanged; only the row-major base advances
// by 8 * 256 half elements.
__device__ __forceinline__ void issue_cp_async_B8(half *smem_B, const half *gB,
                                                  int tB_row, int tB_col,
                                                  int strideB) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kSmemRowStep = 8 * 256;
  int col = tB_col * kElementsPerAccess;
  int smem_base = n256_splitk::offset_B(col, tB_row);
  const char *gB_bytes = reinterpret_cast<const char *>(gB);
  int gmem_base_bytes = (tB_row * strideB + col) * sizeof(half);
  int gmem_row_step_bytes = 8 * strideB * sizeof(half);

  cp_async::cg<16>(&smem_B[smem_base + 0 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 0 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 1 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 1 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 2 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 2 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 3 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 3 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 4 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 4 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 5 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 5 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 6 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 6 * gmem_row_step_bytes);
  cp_async::cg<16>(&smem_B[smem_base + 7 * kSmemRowStep],
                   gB_bytes + gmem_base_bytes + 7 * gmem_row_step_bytes);
}

template <int FirstRowBlock>
__device__ __forceinline__ void issue_cp_async_A2(half *smem_A,
                                                  const half *gA, int tA_row,
                                                  int tA_col, int strideA) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kRowStep = 32;
  int col = tA_col * kElementsPerAccess;
  int row = tA_row + FirstRowBlock * kRowStep;
  int smem_base = hgemm_smem::offset_A(row, col);
  int gmem_base = row * strideA + col;
  int smem_step = kRowStep * 64;
  int gmem_step = kRowStep * strideA;

  cp_async::cg<16>(&smem_A[smem_base], &gA[gmem_base]);
  cp_async::cg<16>(&smem_A[smem_base + smem_step],
                   &gA[gmem_base + gmem_step]);
}

template <int FirstRowBlock>
__device__ __forceinline__ void issue_cp_async_B2(half *smem_B,
                                                  const half *gB, int tB_row,
                                                  int tB_col, int strideB) {
  constexpr int kElementsPerAccess = 8;
  constexpr int kRowStep = 8;
  int col = tB_col * kElementsPerAccess;
  int row = tB_row + FirstRowBlock * kRowStep;
  int smem_base = offset_B(col, row);
  int gmem_base = row * strideB + col;
  int smem_step = kRowStep * 256;
  int gmem_step = kRowStep * strideB;

  cp_async::cg<16>(&smem_B[smem_base], &gB[gmem_base]);
  cp_async::cg<16>(&smem_B[smem_base + smem_step],
                   &gB[gmem_base + gmem_step]);
}

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_B(half *smem_B, const half *gB,
                                                 int tB_row, int tB_col,
                                                 int strideB) {
  constexpr int kElementsPerAccess = 8;
  int row = tB_row + RowBlock * 8;
  int col = tB_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_B[n256_splitk::offset_B(col, row)], &gB[row * strideB + col]);
}

template <int WaitGroup, bool AdvanceGmem, typename Shape_MNK, int kStages>
__device__ __forceinline__ void run_mma_tile_n256(
    float (&tCrC)[4][4][2][2][2],
    half (&tCrA)[2][4][2][2][2],
    half (&tCrB)[2][4][2][2][2],
    HgemmSharedStorage<Shape_MNK, kStages> *smem, int &smem_read_offset,
    int &smem_write_offset, const half *&gA_next, const half *&gB_next,
    int StrideA, int StrideB,
    int tA_row, int tA_col, int tB_row, int tB_col, int warp_m_id,
    int warp_n_id, int ldsmx4_row, int ldsmx4_col, int ldsmx4T_row,
    int ldsmx4T_col) {
  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 64;
  constexpr int Tiled_MMA_K = 16;
  constexpr int K_BLOCK_MAX = 4;
  constexpr int kBufferBytes = sizeof(Buffer<Shape_MNK>);
  constexpr int kAElements = Shape_MNK::M * Shape_MNK::K;
  static_assert(WaitGroup == 0 || WaitGroup == 1,
                "n256 pipeline only uses wait_group 0 or 1");

  // Keep the ring-buffer stage addresses as byte offsets.  Indexing
  // smem->buffer[pipe] makes ptxas materialize pipe * sizeof(Buffer) in the
  // steady-state loop; kBufferBytes is 0xc000 for the 128x256 tile.
  char *smem_bytes = reinterpret_cast<char *>(smem);
  half *smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
  half *smem_read_B = smem_read_A + kAElements;
  half *smem_write_A =
      reinterpret_cast<half *>(smem_bytes + smem_write_offset);
  half *smem_write_B = smem_write_A + kAElements;

  #pragma unroll
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    int k_block_next = (k_block + 1) % K_BLOCK_MAX;
    int k_block_slot = k_block & 1;
    int k_block_next_slot = k_block_next & 1;
    half *b_smem = smem_read_B;
    int b_ldsm_base = n256_splitk::offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32,
        ldsmx4T_col + k_block_next * Tiled_MMA_K);
    if (k_block == 0) {
      issue_cp_async_A4(smem_write_A, gA_next, tA_row, tA_col, StrideA);
      issue_cp_async_B8(smem_write_B, gB_next, tB_row, tB_col, StrideB);
      if constexpr (AdvanceGmem) {
        gA_next += Shape_MNK::K;
        gB_next += Shape_MNK::K * StrideB;
      }
    }
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][0][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][0][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][0][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][0][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][0][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][0][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][0][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][0][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 0]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][1][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][1][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][1][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][1][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 1]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][2][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][2][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][2][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][2][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 2]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[k_block_next_slot][3][0][0][0]),
                      as_u32(tCrB[k_block_next_slot][3][0][1][0]),
                      as_u32(tCrB[k_block_next_slot][3][1][0][0]),
                      as_u32(tCrB[k_block_next_slot][3][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 3]);

    MMA_1_ROW_SLOT_FP32_SWAPPED(0, k_block_slot);
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][1][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][1][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][1][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][1][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    MMA_1_ROW_SLOT_FP32_SWAPPED_REV(1, k_block_slot);
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][2][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][2][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][2][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][2][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    MMA_1_ROW_SLOT_FP32_SWAPPED(2, k_block_slot);
    ldsm::x4<ldsm::N>(as_u32(tCrA[k_block_next_slot][3][0][0][0]),
                      as_u32(tCrA[k_block_next_slot][3][0][1][0]),
                      as_u32(tCrA[k_block_next_slot][3][1][0][0]),
                      as_u32(tCrA[k_block_next_slot][3][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          k_block_next * Tiled_MMA_K + ldsmx4_col * 8));
    MMA_1_ROW_SLOT_FP32_SWAPPED_REV(3, k_block_slot);
    if (k_block == K_BLOCK_MAX - 2) {
      cp_async::commit_group();
      smem_write_offset = smem_read_offset;
      smem_read_offset =
          (smem_read_offset == (kStages - 1) * kBufferBytes)
              ? 0
              : smem_read_offset + kBufferBytes;
      smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
      smem_read_B = smem_read_A + kAElements;
      smem_write_A = reinterpret_cast<half *>(smem_bytes + smem_write_offset);
      smem_write_B = smem_write_A + kAElements;
      cp_async::wait_group<WaitGroup>();
      __syncthreads();
    }
  }
}


__device__ __forceinline__ void store_f32x2(float *sC, int row, int col,
                                            int stride, float x0, float x1) {
  float *ptr = sC + row * stride + col;
  ptr[0] = x0;
  ptr[1] = x1;
}

template <int kStoreIterations, int kCtaN, int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_f32_recurrent(float *gC,
                                                         const float *sC,
                                                         int strideC) {
  constexpr int kElementsPerAccess = 4; // four fp32 values, 16B
  constexpr int kVecsPerRow = kCtaN / kElementsPerAccess;
  constexpr int kRowsPerStep = kThreads / kVecsPerRow;

  int vec_row = threadIdx.x / kVecsPerRow;
  int vec_col = threadIdx.x % kVecsPerRow;
  float *d_ptr = gC + vec_row * strideC + vec_col * kElementsPerAccess;
  const float *s_ptr = sC + vec_row * kSmemStrideC +
                      vec_col * kElementsPerAccess;
  int d_step = kRowsPerStep * strideC;
  constexpr int s_step = kRowsPerStep * kSmemStrideC;

#pragma unroll
  for (int i = 0; i < kStoreIterations; ++i) {
    *reinterpret_cast<uint4 *>(d_ptr) =
        *reinterpret_cast<const uint4 *>(s_ptr);
    d_ptr += d_step;
    s_ptr += s_step;
  }
}

template <typename Shape_MNK = shape_mnk_n256, int kStages, int kBlockSwizzle>
__global__ void sm80_hgemm_f16_nn_m128n256k64_streamk_splitk_kernel(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k) {
  constexpr int kCtaM = Shape_MNK::M; // 128
  constexpr int kCtaN = Shape_MNK::N; // 256
  constexpr int kCtaK = Shape_MNK::K; // 64
  static_assert(kCtaM == 128 && kCtaN == 256 && kCtaK == 64,
                "swizzled shared-memory layout assumes a 128x256x64 CTA");

  constexpr int kWarpsM = 2;
  constexpr int kWarpSize = 32;

  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 64;
  constexpr int Tiled_MMA_K = 16;

  extern __shared__ char shared_memory[];
  using MainLoopSharedStorage = HgemmSharedStorage<Shape_MNK, kStages>;
  MainLoopSharedStorage *smem =
      reinterpret_cast<MainLoopSharedStorage *>(shared_memory);

  int StrideA = K;
  int StrideB = N;

  int const tile_m_max = (M + kCtaM - 1) / kCtaM;
  int const tile_n_max = (N + kCtaN - 1) / kCtaN;

  int tile_m = blockIdx.x / kBlockSwizzle;
  int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tile_m >= tile_m_max || tile_n >= tile_n_max) {
    return;
  }

  if (split_k <= 0 || K % kCtaK != 0) {
    return;
  }
  int const split_id = static_cast<int>(blockIdx.z);
  int const k_tile_count = K / kCtaK;
  if (split_id >= split_k || k_tile_count % split_k != 0) {
    return;
  }
  int const k_tiles_per_split = k_tile_count / split_k;
  int const k_begin = split_id * k_tiles_per_split * kCtaK;

  const half *gA_base =
      A + tile_m * kCtaM * StrideA + k_begin;
  const half *gB_base =
      B + k_begin * StrideB + tile_n * kCtaN;

  long long const matrix_elements = static_cast<long long>(M) * N;
  float *gPartial =
      partial + static_cast<long long>(split_id) * matrix_elements +
      static_cast<long long>(tile_m) * kCtaM * N +
      static_cast<long long>(tile_n) * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;

  int const K_TILE_MAX = k_tiles_per_split;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;
  static_assert(K_BLOCK_MAX == 4,
                "mainloop cp.async schedule assumes four MMA K-blocks");

  constexpr int MMA_M = kCtaM / Tiled_MMA_M;
  constexpr int MMA_N = kCtaN / Tiled_MMA_N;
  constexpr int MMA_K = kCtaK / Tiled_MMA_K;
  constexpr int kFragmentSlots = 2;
  static_assert(MMA_K == K_BLOCK_MAX,
                "fragment slots assume one logical slot per MMA K-block");

  constexpr int Fragment = 2;
  constexpr int CoreMatrix_M = 2;
  constexpr int CoreMatrix_N = 2;
  constexpr int CoreMatrix_K = 2;

  constexpr int kElementsPerAccess = 8; // half, 16B

  // (MMA_M, MMA_N, CoreMatrix_N, CoreMatrix_M, Fragment)
  // :
  // (8 * MMA_N, 8, 4, 2, 1)

  float tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][Fragment];
  // Put the double-buffer slot first so each slot's fragment registers are
  // contiguous in the local array layout.
  half tCrA[kFragmentSlots][MMA_M][CoreMatrix_K][CoreMatrix_M][Fragment];
  half tCrB[kFragmentSlots][MMA_N][CoreMatrix_N][CoreMatrix_K][Fragment];

  int lane_id = tid % kWarpSize;

  int tA_row = tid / (kCtaK / kElementsPerAccess); // 8
  int tA_col = tid % (kCtaK / kElementsPerAccess);

  int tB_row = tid / (kCtaN / kElementsPerAccess); // 32
  int tB_col = tid % (kCtaN / kElementsPerAccess);

  int k_tiles_to_compute = K_TILE_MAX;
  const half *gA_next = gA_base;
  const half *gB_next = gB_base;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    issue_cp_async_A4(smem->buffer[k_pipe].A, gA_next, tA_row, tA_col,
                      StrideA);
    issue_cp_async_B8(smem->buffer[k_pipe].B, gB_next, tB_row, tB_col,
                      StrideB);

    cp_async::commit_group();
    --k_tiles_to_compute;
    if (k_tiles_to_compute > 0) {
      gA_next += kCtaK;
      gB_next += kCtaK * StrideB;
    }
  }

  constexpr int kBufferBytes = sizeof(Buffer<Shape_MNK>);
  constexpr int kAElements = Shape_MNK::M * Shape_MNK::K;
  int smem_read_offset = 0;
  int smem_write_offset = (K_PIPE_MAX - 1) * kBufferBytes;
  char *smem_bytes = reinterpret_cast<char *>(smem);
  half *smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
  half *smem_read_B = smem_read_A + kAElements;

  int warp_m_id = warp_id % kWarpsM;
  int warp_n_id = warp_id / kWarpsM;

  int ldsmx4_row = lane_id % 16;
  int ldsmx4_col = lane_id / 16;

  int ldsmx4T_col = lane_id % 16;
  int ldsmx4T_row = lane_id / 16;

  if constexpr (K_BLOCK_MAX > 1) {
    cp_async::wait_group<K_PIPE_MAX - 2>();
    __syncthreads();

    ldsm::x4<ldsm::N>(as_u32(tCrA[0][0][0][0][0]), as_u32(tCrA[0][0][0][1][0]),
                      as_u32(tCrA[0][0][1][0][0]), as_u32(tCrA[0][0][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][1][0][0][0]), as_u32(tCrA[0][1][0][1][0]),
                      as_u32(tCrA[0][1][1][0][0]), as_u32(tCrA[0][1][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][2][0][0][0]), as_u32(tCrA[0][2][0][1][0]),
                      as_u32(tCrA[0][2][1][0][0]), as_u32(tCrA[0][2][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][3][0][0][0]), as_u32(tCrA[0][3][0][1][0]),
                      as_u32(tCrA[0][3][1][0][0]), as_u32(tCrA[0][3][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));

    // The four B fragments are separated by Tiled_MMA_N columns.  For this
    // swizzle, n + 64 changes the half-element offset by exactly 64, so the
    // four LDSM instructions can share one computed base address.
    half *b_smem = smem_read_B;
    int b_ldsm_base = n256_splitk::offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32,
        ldsmx4T_col + 0 * Tiled_MMA_K);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 0]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][1][0][0][0]), as_u32(tCrB[0][1][0][1][0]),
                      as_u32(tCrB[0][1][1][0][0]), as_u32(tCrB[0][1][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 1]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][2][0][0][0]), as_u32(tCrB[0][2][0][1][0]),
                      as_u32(tCrB[0][2][1][0][0]), as_u32(tCrB[0][2][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 2]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][3][0][0][0]), as_u32(tCrB[0][3][0][1][0]),
                      as_u32(tCrB[0][3][1][0][0]), as_u32(tCrB[0][3][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 3]);
  }
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cm_n = 0; cm_n < CoreMatrix_N; ++cm_n) {
#pragma unroll
        for (int cm_m = 0; cm_m < CoreMatrix_M; ++cm_m) {
          tCrC[m][n][cm_n][cm_m][0] = 0.0f;
          tCrC[m][n][cm_n][cm_m][1] = 0.0f;
        }
      }
    }
  }

  // Match CUTLASS 3's SM80 unpredicated mainloop.  The prologue already
  // consumed K_PIPE_MAX - 1 tiles, so the loop computes exactly K_TILE_MAX
  // tiles while running K_PIPE_MAX - 1 extra drain iterations in its copy
  // schedule.  Once the last real tile is selected, run_mma_tile_n256 keeps
  // reissuing that valid tile instead of generating predicated or OOB copies.
#pragma unroll 1
  for (; k_tiles_to_compute > 1; --k_tiles_to_compute) {
    run_mma_tile_n256<K_PIPE_MAX - 2, true>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, StrideA, StrideB, tA_row, tA_col, tB_row, tB_col, warp_m_id,
        warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row, ldsmx4T_col);
  }

#pragma unroll
  for (; k_tiles_to_compute > -(K_PIPE_MAX - 1); --k_tiles_to_compute) {
    run_mma_tile_n256<K_PIPE_MAX - 2, false>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, StrideA, StrideB, tA_row, tA_col, tB_row, tB_col, warp_m_id,
        warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row, ldsmx4T_col);
  }

  //
  // Epilogue
  //

  cp_async::wait_all();
  __syncthreads();
  float *sC = reinterpret_cast<float *>(shared_memory);

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 264;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][0][0], tCrC[m][n][0][0][1]);
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][1][0], tCrC[m][n][0][1][1]);
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][0][0], tCrC[m][n][1][0][1]);
      store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][1][0], tCrC[m][n][1][1][1]);
    }
  }

  __syncthreads();

  constexpr int kEpilogueThreads = 256;
  constexpr int kEpilogueElementsPerAccess = 4;
  constexpr int kEpilogueVecCount =
      kCtaM * kCtaN / kEpilogueElementsPerAccess;
  static_assert(kEpilogueVecCount % kEpilogueThreads == 0,
                "epilogue store schedule assumes full fixed-thread coverage");
  constexpr int kEpilogueStoreIterations =
      kEpilogueVecCount / kEpilogueThreads;
  store_gmem_f32_recurrent<kEpilogueStoreIterations, kCtaN,
                           kEpilogueThreads, kSmemStrideC>(
      gPartial, sC, N);
}



constexpr int kSplitKStages = 3;
constexpr int kSplitKThreads = 256;
constexpr int kSplitKSharedStorageBytes128x256 =
    sizeof(HgemmSharedStorage<shape_mnk_n256, kSplitKStages>);

template <int BlockSwizzle>
inline cudaError_t configure_hgemm_128x256_splitk_fp32acc() {
  auto kernel_fptr =
      sm80_hgemm_f16_nn_m128n256k64_streamk_splitk_kernel<
          shape_mnk_n256, kSplitKStages, BlockSwizzle>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSplitKSharedStorageBytes128x256);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaFuncSetAttribute(kernel_fptr,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              100);
}

inline cudaError_t configure_hgemm_128x256_splitk_fp32acc(int block_swizzle) {
  switch (block_swizzle) {
  case 1:
    return configure_hgemm_128x256_splitk_fp32acc<1>();
  case 2:
    return configure_hgemm_128x256_splitk_fp32acc<2>();
  case 4:
    return configure_hgemm_128x256_splitk_fp32acc<4>();
  case 8:
    return configure_hgemm_128x256_splitk_fp32acc<8>();
  case 16:
    return configure_hgemm_128x256_splitk_fp32acc<16>();
  case 32:
    return configure_hgemm_128x256_splitk_fp32acc<32>();
  case 64:
    return configure_hgemm_128x256_splitk_fp32acc<64>();
  default:
    return cudaErrorInvalidValue;
  }
}

template <int BlockSwizzle>
inline void launch_hgemm_128x256_splitk_fp32acc_unchecked(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k, cudaStream_t stream = 0) {
  int tile_m_count = M / shape_mnk_n256::M;
  int tile_n_count = N / shape_mnk_n256::N;
  dim3 block(kSplitKThreads);
  dim3 grid(tile_m_count * BlockSwizzle,
            (tile_n_count + BlockSwizzle - 1) / BlockSwizzle, split_k);
  sm80_hgemm_f16_nn_m128n256k64_streamk_splitk_kernel<
      shape_mnk_n256, kSplitKStages, BlockSwizzle>
      <<<grid, block, kSplitKSharedStorageBytes128x256, stream>>>(
          A, B, partial, M, N, K, split_k);
}

inline void launch_hgemm_128x256_splitk_fp32acc_unchecked(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k, int block_swizzle, cudaStream_t stream = 0) {
  switch (block_swizzle) {
  case 1:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<1>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 2:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<2>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 4:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<4>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 8:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<8>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 16:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<16>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 32:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<32>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  case 64:
    launch_hgemm_128x256_splitk_fp32acc_unchecked<64>(
        A, B, partial, M, N, K, split_k, stream);
    return;
  default:
    return;
  }
}

inline cudaError_t launch_hgemm_128x256_splitk_fp32acc(
    const half *A, const half *B, float *partial, int M, int N, int K,
    int split_k, int block_swizzle, cudaStream_t stream = 0) {
  cudaError_t err =
      configure_hgemm_128x256_splitk_fp32acc(block_swizzle);
  if (err != cudaSuccess) {
    return err;
  }
  launch_hgemm_128x256_splitk_fp32acc_unchecked(
      A, B, partial, M, N, K, split_k, block_swizzle, stream);
  return cudaGetLastError();
}

template <int SplitK>
__global__ void sm80_hgemm_f16_nn_m128n256k64_streamk_splitk_reduce_kernel(
    const float *__restrict__ partial, half *__restrict__ C, int M, int N) {
  static_assert(SplitK > 0, "SplitK must be positive");
  constexpr int kTileM = shape_mnk_n256::M;
  constexpr int kTileN = shape_mnk_n256::N;
  constexpr int kThreads = 256;
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerRow = kTileN / kElementsPerVector;
  constexpr int kTileVectors = kTileM * kVectorsPerRow;

  int const tile_m = static_cast<int>(blockIdx.x);
  int const tile_n = static_cast<int>(blockIdx.y);
  int const tid = static_cast<int>(threadIdx.x);
  long long const matrix_elements = static_cast<long long>(M) * N;
  long long const tile_row_base =
      static_cast<long long>(tile_m) * kTileM;
  long long const tile_col_base =
      static_cast<long long>(tile_n) * kTileN;

  for (int vec = tid; vec < kTileVectors; vec += kThreads) {
    int const row = vec / kVectorsPerRow;
    int const col = (vec % kVectorsPerRow) * kElementsPerVector;
    long long const output_index =
        (tile_row_base + row) * N + tile_col_base + col;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma unroll
    for (int split = 0; split < SplitK; ++split) {
      float4 value = *reinterpret_cast<const float4 *>(
          partial + static_cast<long long>(split) * matrix_elements +
          output_index);
      acc0 += value.x;
      acc1 += value.y;
      acc2 += value.z;
      acc3 += value.w;
    }

    half2 out01 = __floats2half2_rn(acc0, acc1);
    half2 out23 = __floats2half2_rn(acc2, acc3);
    half *out = C + output_index;
    *reinterpret_cast<half2 *>(out + 0) = out01;
    *reinterpret_cast<half2 *>(out + 2) = out23;
  }
}

template <int SplitK>
inline cudaError_t launch_hgemm_128x256_splitk_reduce_unchecked(
    const float *partial, half *C, int M, int N,
    cudaStream_t stream = 0) {
  dim3 block(256);
  dim3 grid(M / shape_mnk_n256::M, N / shape_mnk_n256::N);
  sm80_hgemm_f16_nn_m128n256k64_streamk_splitk_reduce_kernel<SplitK>
      <<<grid, block, 0, stream>>>(partial, C, M, N);
  return cudaGetLastError();
}

inline cudaError_t launch_hgemm_128x256_splitk_reduce(
    const float *partial, half *C, int M, int N, int split_k,
    cudaStream_t stream = 0) {
  switch (split_k) {
  case 1:
    return launch_hgemm_128x256_splitk_reduce_unchecked<1>(
        partial, C, M, N, stream);
  case 2:
    return launch_hgemm_128x256_splitk_reduce_unchecked<2>(
        partial, C, M, N, stream);
  case 4:
    return launch_hgemm_128x256_splitk_reduce_unchecked<4>(
        partial, C, M, N, stream);
  case 8:
    return launch_hgemm_128x256_splitk_reduce_unchecked<8>(
        partial, C, M, N, stream);
  case 16:
    return launch_hgemm_128x256_splitk_reduce_unchecked<16>(
        partial, C, M, N, stream);
  case 32:
    return launch_hgemm_128x256_splitk_reduce_unchecked<32>(
        partial, C, M, N, stream);
  case 64:
    return launch_hgemm_128x256_splitk_reduce_unchecked<64>(
        partial, C, M, N, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

} // namespace n256_splitk


#include <stddef.h>
#include <stdint.h>

// A small SM80-oriented subset of the basic Stream-K decomposition.  This
// header provides the work mapping and communication-slot primitives used by
// the 128x256 kernel.  It intentionally does not include cluster scheduling
// or persistent cooperative launch.
namespace indigo::hgemm::sm80::streamk {

constexpr int64_t kInvalidWork = -1;

__host__ __device__ __forceinline__ int64_t ceil_div_i64(int64_t x,
                                                          int64_t y) {
  return x <= 0 ? 0 : (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int64_t min_i64(int64_t x, int64_t y) {
  return x < y ? x : y;
}

// One logical work iteration is one CTA-wide BLK_M x BLK_N x BLK_K MAC
// iteration.  The global work order is row-major over output tiles, followed
// by the K-tile iterations of each output tile.
struct Schedule {
  int problem_m = 0;
  int problem_n = 0;
  int problem_k = 0;
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
  int grid_ctas = 0;

  int tiles_m = 0;
  int tiles_n = 0;
  int tiles_k = 0;
  int64_t tile_count = 0;
  int64_t total_work = 0;
  int64_t work_per_cta = 0;

  __host__ __device__ bool valid() const {
    return problem_m > 0 && problem_n > 0 && problem_k > 0 && tile_m > 0 &&
           tile_n > 0 && tile_k > 0 && grid_ctas > 0 && tiles_m > 0 &&
           tiles_n > 0 && tiles_k > 0 && total_work > 0 && work_per_cta > 0;
  }
};

__host__ __device__ inline Schedule make_schedule(int M, int N, int K,
                                                   int tile_m, int tile_n,
                                                   int tile_k, int grid_ctas) {
  Schedule schedule;
  schedule.problem_m = M;
  schedule.problem_n = N;
  schedule.problem_k = K;
  schedule.tile_m = tile_m;
  schedule.tile_n = tile_n;
  schedule.tile_k = tile_k;
  schedule.grid_ctas = grid_ctas;

  if (M <= 0 || N <= 0 || K <= 0 || tile_m <= 0 || tile_n <= 0 ||
      tile_k <= 0 || grid_ctas <= 0) {
    return schedule;
  }

  schedule.tiles_m = static_cast<int>(ceil_div_i64(M, tile_m));
  schedule.tiles_n = static_cast<int>(ceil_div_i64(N, tile_n));
  schedule.tiles_k = static_cast<int>(ceil_div_i64(K, tile_k));
  schedule.tile_count = static_cast<int64_t>(schedule.tiles_m) *
                        schedule.tiles_n;
  schedule.total_work = schedule.tile_count * schedule.tiles_k;
  schedule.work_per_cta =
      ceil_div_i64(schedule.total_work, grid_ctas);
  return schedule;
}

// Use this when the caller wants a fixed grid width, normally the number of
// SMs or a small multiple of it.  It avoids launching more CTAs than there
// are logical work iterations.
__host__ __device__ inline int choose_grid_ctas(int64_t total_work,
                                                int requested_ctas) {
  if (total_work <= 0) {
    return 0;
  }
  if (requested_ctas <= 0 || requested_ctas > total_work) {
    return total_work > INT32_MAX ? INT32_MAX
                                  : static_cast<int>(total_work);
  }
  return requested_ctas;
}

struct WorkRange {
  int cta_id = -1;
  int64_t begin = 0;
  int64_t end = 0;

  __host__ __device__ bool valid() const {
    return cta_id >= 0 && begin < end;
  }
};

__host__ __device__ inline WorkRange get_work_range(Schedule const &schedule,
                                                    int cta_id) {
  WorkRange range;
  range.cta_id = cta_id;
  if (!schedule.valid() || cta_id < 0 || cta_id >= schedule.grid_ctas) {
    return range;
  }
  range.begin = min_i64(static_cast<int64_t>(cta_id) *
                            schedule.work_per_cta,
                        schedule.total_work);
  range.end = min_i64(range.begin + schedule.work_per_cta,
                      schedule.total_work);
  if (range.begin >= range.end) {
    range.cta_id = -1;
  }
  return range;
}

// A CTA may process several segments.  A segment is the intersection of the
// CTA's contiguous work interval and one output tile's K-iteration interval.
struct TileSegment {
  int tile_id = -1;
  int tile_m = -1;
  int tile_n = -1;
  int k_tile_begin = 0;
  int k_tile_end = 0;
  int k_begin = 0;
  int k_end = 0;
  int64_t next_work = 0;
  bool starts_tile = false;
  bool ends_tile = false;

  __host__ __device__ bool valid() const {
    return tile_id >= 0 && tile_m >= 0 && tile_n >= 0 &&
           k_tile_begin < k_tile_end && k_begin < k_end;
  }
};

__host__ __device__ inline TileSegment get_tile_segment(
    Schedule const &schedule, WorkRange const &range, int64_t work) {
  TileSegment segment;
  if (!schedule.valid() || !range.valid() || work < range.begin ||
      work >= range.end) {
    return segment;
  }

  int64_t tile_id_i64 = work / schedule.tiles_k;
  int64_t tile_work_begin = tile_id_i64 * schedule.tiles_k;
  int64_t segment_work_end = min_i64(
      range.end, tile_work_begin + static_cast<int64_t>(schedule.tiles_k));
  int64_t k_tile_begin = work - tile_work_begin;
  int64_t k_tile_end = segment_work_end - tile_work_begin;

  segment.tile_id = static_cast<int>(tile_id_i64);
  segment.tile_m = segment.tile_id / schedule.tiles_n;
  segment.tile_n = segment.tile_id % schedule.tiles_n;
  segment.k_tile_begin = static_cast<int>(k_tile_begin);
  segment.k_tile_end = static_cast<int>(k_tile_end);
  segment.k_begin = static_cast<int>(k_tile_begin * schedule.tile_k);
  segment.k_end = static_cast<int>(min_i64(
      k_tile_end * static_cast<int64_t>(schedule.tile_k), schedule.problem_k));
  segment.next_work = segment_work_end;
  segment.starts_tile = k_tile_begin == 0;
  segment.ends_tile = k_tile_end == schedule.tiles_k;
  return segment;
}

__host__ __device__ inline int tile_owner_cta(Schedule const &schedule,
                                              int tile_id) {
  if (!schedule.valid() || tile_id < 0 ||
      static_cast<int64_t>(tile_id) >= schedule.tile_count) {
    return -1;
  }
  int64_t tile_begin = static_cast<int64_t>(tile_id) * schedule.tiles_k;
  return static_cast<int>(tile_begin / schedule.work_per_cta);
}

// Returns the exclusive CTA id bound for peer CTAs that may contribute to a
// tile.  The owner should wait on producer IDs [owner + 1, peer_end).
__host__ __device__ inline int tile_peer_cta_end(Schedule const &schedule,
                                                 int tile_id) {
  if (!schedule.valid() || tile_id < 0 ||
      static_cast<int64_t>(tile_id) >= schedule.tile_count) {
    return -1;
  }
  int64_t tile_end =
      min_i64((static_cast<int64_t>(tile_id) + 1) * schedule.tiles_k,
              schedule.total_work);
  int64_t peer_end = ceil_div_i64(tile_end, schedule.work_per_cta);
  return peer_end > schedule.grid_ctas ? schedule.grid_ctas
                                        : static_cast<int>(peer_end);
}

// Number of CTAs whose half-open work ranges intersect this tile.  This count
// includes the first producer CTA returned by tile_owner_cta().
__host__ __device__ inline int tile_producer_count(Schedule const &schedule,
                                                   int tile_id) {
  int first = tile_owner_cta(schedule, tile_id);
  int end = tile_peer_cta_end(schedule, tile_id);
  return first >= 0 && end > first ? end - first : 0;
}

// Basic Stream-K uses one temporary partial slot per CTA.  A CTA can have at
// most one non-owner segment: after publishing that leading partial, all
// later complete/ending segments are handled by the CTA that starts the tile.
struct PartialWorkspace {
  float *partials = nullptr; // [grid_ctas][tile_m * tile_n], FP32
  int *flags = nullptr;      // [grid_ctas], zero-initialized before launch
  int64_t tile_elements = 0;

  __device__ __forceinline__ float *slot(int cta_id) const {
    return partials + static_cast<int64_t>(cta_id) * tile_elements;
  }
};

__host__ __device__ inline int64_t partial_workspace_elements(
    Schedule const &schedule) {
  return static_cast<int64_t>(schedule.grid_ctas) * schedule.tile_m *
         schedule.tile_n;
}

// All CTA threads call this after storing their partial tile.  The block
// barrier makes the stores complete within the CTA; the fence then publishes
// them before the flag becomes visible to the owner CTA.
__device__ __forceinline__ void publish_partial(int *flags, int cta_id) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicExch(flags + cta_id, 1);
  }
}

// Flags must be reset to zero before the kernel launch.  This is an acquire-
// style polling helper for the owner CTA.
__device__ __forceinline__ void wait_partial(int *flags, int cta_id) {
  while (atomicAdd(flags + cta_id, 0) == 0) {
    __nanosleep(64);
  }
  __threadfence();
}

} // namespace indigo::hgemm::sm80::streamk


#include <stddef.h>
#include <stdint.h>

// A simple SM80 Stream-K implementation for the existing 128x256 mainloop.
//
// The logical scheduler K tile is 128 elements.  The existing optimized
// mainloop consumes two 64-wide K tiles per segment, which keeps its cp.async
// prologue/drain contract intact without changing the mainloop itself.
namespace n256_streamk {

namespace sk = indigo::hgemm::sm80::streamk;

constexpr int kStreamKTileM = 128;
constexpr int kStreamKTileN = 256;
constexpr int kStreamKTileK = 128;
constexpr int kMainloopTileK = 64;
constexpr int kStreamKThreads = 256;
constexpr int kStreamKStages = 3;
constexpr int kMinStreamKIterations = 8;
constexpr int kStreamKTileElements = kStreamKTileM * kStreamKTileN;
constexpr int kStreamKSharedStorageBytes128x256 =
    sizeof(HgemmSharedStorage<shape_mnk_n256, kStreamKStages>);

// Host-side decomposition and workspace plan.  The terminology follows
// CUTLASS's ThreadblockSwizzleStreamK: tiled_shape_* describes the output
// tile grid, dp_blocks are direct-output CTAs, sk_tiles are output tiles
// covered by Stream-K, and sk_blocks are Stream-K CTAs.
struct StreamKSchedulePlan {
  int tiled_shape_m = 0;
  int tiled_shape_n = 0;
  int iters_per_tile = 0;
  int output_tile_count = 0;

  int dp_blocks = 0;
  int sk_tiles = 0;
  int sk_blocks = 0;
  int64_t sk_iters_per_block = 0;
  int max_peers_per_tile = 0;
  size_t partials_elements = 0;

  bool valid() const {
    if (tiled_shape_m <= 0 || tiled_shape_n <= 0 ||
        iters_per_tile <= 0 || output_tile_count <= 0 || dp_blocks < 0 ||
        sk_tiles < 0 || dp_blocks + sk_tiles != output_tile_count) {
      return false;
    }
    if (sk_tiles == 0) {
      return dp_blocks == output_tile_count;
    }
    return sk_blocks > 0 && sk_iters_per_block > 0 &&
           max_peers_per_tile > 0 && partials_elements > 0;
  }
};

// Device-side scheduler parameters.  This is the small Params object passed
// to both kernels, analogous to CUTLASS's kernel Params/block-mapping state.
// All shape-derived and decomposition-derived values are prepared on host.
struct StreamKParams {
  int tiled_shape_n = 0;
  int iters_per_tile = 0;
  int dp_blocks = -1;
  int sk_tiles = 0;
  int sk_blocks = 0;
  int64_t sk_iters_per_block = 0;
  int max_peers_per_tile = 0;

  bool valid() const {
    if (tiled_shape_n <= 0 || iters_per_tile <= 0 || dp_blocks < 0 ||
        sk_tiles < 0) {
      return false;
    }
    if (sk_tiles == 0) {
      return true;
    }
    return sk_blocks > 0 && sk_iters_per_block > 0 &&
           max_peers_per_tile > 0;
  }
};

// The split factor is an upper bound on the requested Stream-K grid width.
// The final grid is also bounded by the device wave size and CUTLASS's
// minimum useful K-iteration count per Stream-K unit. It is not a fixed K
// partition: the final CTA ranges are assigned by the linear decomposition.
__host__ __device__ inline int get_tile_peer_count(
    int tile_id, int sk_tiles, int iters_per_tile, int sk_blocks,
    int64_t sk_iters_per_block) {
  if (tile_id < 0 || tile_id >= sk_tiles || iters_per_tile <= 0 ||
      sk_blocks <= 0 || sk_iters_per_block <= 0) {
    return 0;
  }
  int64_t tile_begin = static_cast<int64_t>(tile_id) * iters_per_tile;
  int64_t tile_end = tile_begin + iters_per_tile;
  int first = static_cast<int>(tile_begin / sk_iters_per_block);
  int end = static_cast<int>((tile_end + sk_iters_per_block - 1) /
                             sk_iters_per_block);
  end = end > sk_blocks ? sk_blocks : end;
  return end > first ? end - first : 0;
}

inline StreamKSchedulePlan make_streamk_schedule_plan(
    int M, int N, int K, int split_k_factor, int sm_count) {
  StreamKSchedulePlan plan;
  if (M <= 0 || N <= 0 || K <= 0 || M % kStreamKTileM != 0 ||
      N % kStreamKTileN != 0 || K % kStreamKTileK != 0 ||
      split_k_factor <= 0 ||
      sm_count <= 0) {
    return plan;
  }

  plan.tiled_shape_m = M / kStreamKTileM;
  plan.tiled_shape_n = N / kStreamKTileN;
  plan.iters_per_tile = K / kStreamKTileK;
  int64_t output_tile_count_i64 =
      static_cast<int64_t>(plan.tiled_shape_m) * plan.tiled_shape_n;
  if (output_tile_count_i64 > INT32_MAX) {
    return StreamKSchedulePlan{};
  }
  plan.output_tile_count = static_cast<int>(output_tile_count_i64);

  int64_t full_waves = output_tile_count_i64 / sm_count;
  int64_t total_waves =
      (output_tile_count_i64 + static_cast<int64_t>(sm_count) - 1) /
      sm_count;
  int64_t dp_blocks_i64 = 0;
  if (full_waves != total_waves &&
      plan.iters_per_tile > kMinStreamKIterations) {
    int64_t dp_waves = full_waves > 1 ? full_waves - 1 : 0;
    dp_blocks_i64 = dp_waves * sm_count;
  }
  plan.dp_blocks = static_cast<int>(dp_blocks_i64);
  plan.sk_tiles = plan.output_tile_count - plan.dp_blocks;
  if (plan.sk_tiles == 0) {
    return plan;
  }

  int64_t requested_ctas_i64 =
      static_cast<int64_t>(plan.sk_tiles) * split_k_factor;
  int requested_ctas = requested_ctas_i64 > INT32_MAX
                           ? INT32_MAX
                           : static_cast<int>(requested_ctas_i64);
  int64_t total_work =
      static_cast<int64_t>(plan.sk_tiles) * plan.iters_per_tile;
  int64_t min_sized_ctas = total_work / kMinStreamKIterations;
  int64_t target_ctas = min_sized_ctas < sm_count ? min_sized_ctas : sm_count;
  plan.sk_blocks = requested_ctas < target_ctas
                       ? requested_ctas
                       : static_cast<int>(target_ctas);
  if (plan.sk_blocks <= 0) {
    return StreamKSchedulePlan{};
  }

  plan.sk_iters_per_block =
      (total_work + static_cast<int64_t>(plan.sk_blocks) - 1) /
      plan.sk_blocks;

  for (int tile_id = 0; tile_id < plan.sk_tiles; ++tile_id) {
    int peers = get_tile_peer_count(
        tile_id, plan.sk_tiles, plan.iters_per_tile, plan.sk_blocks,
        plan.sk_iters_per_block);
    if (peers <= 0) {
      return StreamKSchedulePlan{};
    }
    if (peers > plan.max_peers_per_tile) {
      plan.max_peers_per_tile = peers;
    }
  }

  uint64_t per_tile = static_cast<uint64_t>(plan.max_peers_per_tile) *
                      static_cast<uint64_t>(kStreamKTileElements);
  uint64_t total_elements = static_cast<uint64_t>(plan.sk_tiles) * per_tile;
  if (total_elements > static_cast<uint64_t>(SIZE_MAX)) {
    return StreamKSchedulePlan{};
  }
  plan.partials_elements = static_cast<size_t>(total_elements);
  return plan;
}

inline StreamKParams make_streamk_params(StreamKSchedulePlan const &plan) {
  StreamKParams params;
  if (!plan.valid()) {
    return params;
  }

  params.tiled_shape_n = plan.tiled_shape_n;
  params.iters_per_tile = plan.iters_per_tile;
  params.dp_blocks = plan.dp_blocks;
  params.sk_tiles = plan.sk_tiles;
  params.sk_blocks = plan.sk_blocks;
  params.sk_iters_per_block = plan.sk_iters_per_block;
  params.max_peers_per_tile = plan.max_peers_per_tile;
  return params.valid() ? params : StreamKParams{};
}

// DP CTAs use the same FP32 accumulator layout as Stream-K, but convert the
// final tile directly to the real C tensor instead of publishing a partial.
// Each thread owns four FP32 values, which become two half2 stores.
template <int kStoreIterations, int kCtaN, int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_f16_from_f32_recurrent(
    half *gC, const float *sC, int strideC) {
  constexpr int kElementsPerAccess = 4;
  constexpr int kVecsPerRow = kCtaN / kElementsPerAccess;
  constexpr int kRowsPerStep = kThreads / kVecsPerRow;

  int vec_row = threadIdx.x / kVecsPerRow;
  int vec_col = threadIdx.x % kVecsPerRow;
  half *d_ptr = gC + vec_row * strideC + vec_col * kElementsPerAccess;
  const float *s_ptr = sC + vec_row * kSmemStrideC +
                       vec_col * kElementsPerAccess;
  int d_step = kRowsPerStep * strideC;
  constexpr int s_step = kRowsPerStep * kSmemStrideC;

#pragma unroll
  for (int i = 0; i < kStoreIterations; ++i) {
    float4 value = *reinterpret_cast<const float4 *>(s_ptr);
    half2 out01 = __floats2half2_rn(value.x, value.y);
    half2 out23 = __floats2half2_rn(value.z, value.w);
    *reinterpret_cast<half2 *>(d_ptr + 0) = out01;
    *reinterpret_cast<half2 *>(d_ptr + 2) = out23;
    d_ptr += d_step;
    s_ptr += s_step;
  }
}

template <int kStages>
__device__ __forceinline__ void compute_and_store_segment(
    const half *gA_base, const half *gB_base, char *gOutput,
    HgemmSharedStorage<shape_mnk_n256, kStages> *smem,
    char *shared_memory, int stride_A, int stride_B,
    int k_tiles_to_compute, bool store_to_output) {
  constexpr int kCtaM = shape_mnk_n256::M;
  constexpr int kCtaN = shape_mnk_n256::N;
  constexpr int kCtaK = shape_mnk_n256::K;
  constexpr int kWarpsM = 2;
  constexpr int kWarpSize = 32;
  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 64;
  constexpr int Tiled_MMA_K = 16;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;
  constexpr int MMA_M = kCtaM / Tiled_MMA_M;
  constexpr int MMA_N = kCtaN / Tiled_MMA_N;
  constexpr int MMA_K = kCtaK / Tiled_MMA_K;
  constexpr int kFragmentSlots = 2;
  constexpr int CoreMatrix_M = 2;
  constexpr int CoreMatrix_N = 2;
  constexpr int CoreMatrix_K = 2;
  constexpr int kElementsPerAccess = 8;

  static_assert(kCtaM == 128 && kCtaN == 256 && kCtaK == 64,
                "Stream-K mainloop assumes a 128x256x64 CTA");
  static_assert(K_BLOCK_MAX == 4 && MMA_K == K_BLOCK_MAX,
                "Stream-K fragment layout assumes four MMA K-blocks");
  static_assert(K_PIPE_MAX == 3,
                "Stream-K wrapper currently uses the three-stage mainloop");

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane_id = tid % kWarpSize;

  int tA_row = tid / (kCtaK / kElementsPerAccess);
  int tA_col = tid % (kCtaK / kElementsPerAccess);
  int tB_row = tid / (kCtaN / kElementsPerAccess);
  int tB_col = tid % (kCtaN / kElementsPerAccess);

  int warp_m_id = warp_id % kWarpsM;
  int warp_n_id = warp_id / kWarpsM;
  int ldsmx4_row = lane_id % 16;
  int ldsmx4_col = lane_id / 16;
  int ldsmx4T_col = lane_id % 16;
  int ldsmx4T_row = lane_id / 16;

  float tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][2];
  half tCrA[kFragmentSlots][MMA_M][CoreMatrix_K][CoreMatrix_M][2];
  half tCrB[kFragmentSlots][MMA_N][CoreMatrix_N][CoreMatrix_K][2];

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cm_n = 0; cm_n < CoreMatrix_N; ++cm_n) {
#pragma unroll
        for (int cm_m = 0; cm_m < CoreMatrix_M; ++cm_m) {
          tCrC[m][n][cm_n][cm_m][0] = 0.0f;
          tCrC[m][n][cm_n][cm_m][1] = 0.0f;
        }
      }
    }
  }

  // The scheduler uses 128-wide logical K work units, so every valid segment
  // contains at least two 64-wide mainloop tiles.
  if (k_tiles_to_compute < 2) {
    return;
  }

  // Keep the two-stage prologue explicit.  Besides matching the fixed
  // three-stage pipeline contract, this prevents the next-GEMM pointers from
  // being live across the whole prologue and the first LDSM group.
  n256_splitk::issue_cp_async_A4(smem->buffer[0].A, gA_base, tA_row, tA_col,
                                 stride_A);
  n256_splitk::issue_cp_async_B8(smem->buffer[0].B, gB_base, tB_row, tB_col,
                                 stride_B);
  cp_async::commit_group();
  --k_tiles_to_compute;

  n256_splitk::issue_cp_async_A4(smem->buffer[1].A, gA_base + kCtaK, tA_row,
                                 tA_col, stride_A);
  n256_splitk::issue_cp_async_B8(
      smem->buffer[1].B, gB_base + kCtaK * stride_B, tB_row, tB_col,
      stride_B);
  cp_async::commit_group();
  --k_tiles_to_compute;

  // The existing mainloop reissues the last valid tile during its drain.  If
  // real tiles remain, the pointer starts at the first not-yet-issued tile;
  // otherwise it points at the second prologue tile.
  const int next_k_offset = (k_tiles_to_compute > 0) ? 2 * kCtaK : kCtaK;
  const half *gA_next = gA_base + next_k_offset;
  const half *gB_next = gB_base + next_k_offset * stride_B;

  constexpr int kBufferBytes = sizeof(Buffer<shape_mnk_n256>);
  constexpr int kAElements = shape_mnk_n256::M * shape_mnk_n256::K;
  int smem_read_offset = 0;
  int smem_write_offset = (K_PIPE_MAX - 1) * kBufferBytes;
  char *smem_bytes = reinterpret_cast<char *>(smem);
  half *smem_read_A = reinterpret_cast<half *>(smem_bytes + smem_read_offset);
  half *smem_read_B = smem_read_A + kAElements;

  if constexpr (K_BLOCK_MAX > 1) {
    cp_async::wait_group<K_PIPE_MAX - 2>();
    __syncthreads();

    ldsm::x4<ldsm::N>(as_u32(tCrA[0][0][0][0][0]),
                      as_u32(tCrA[0][0][0][1][0]),
                      as_u32(tCrA[0][0][1][0][0]),
                      as_u32(tCrA[0][0][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][1][0][0][0]),
                      as_u32(tCrA[0][1][0][1][0]),
                      as_u32(tCrA[0][1][1][0][0]),
                      as_u32(tCrA[0][1][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][2][0][0][0]),
                      as_u32(tCrA[0][2][0][1][0]),
                      as_u32(tCrA[0][2][1][0][0]),
                      as_u32(tCrA[0][2][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));
    ldsm::x4<ldsm::N>(as_u32(tCrA[0][3][0][0][0]),
                      as_u32(tCrA[0][3][0][1][0]),
                      as_u32(tCrA[0][3][1][0][0]),
                      as_u32(tCrA[0][3][1][1][0]),
                      smem_read_A + hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8));

    half *b_smem = smem_read_B;
    int b_ldsm_base = n256_splitk::offset_B(
        warp_n_id * 8 + ldsmx4T_row * 32, ldsmx4T_col);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]),
                      as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]),
                      as_u32(tCrB[0][0][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 0]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][1][0][0][0]),
                      as_u32(tCrB[0][1][0][1][0]),
                      as_u32(tCrB[0][1][1][0][0]),
                      as_u32(tCrB[0][1][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 1]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][2][0][0][0]),
                      as_u32(tCrB[0][2][0][1][0]),
                      as_u32(tCrB[0][2][1][0][0]),
                      as_u32(tCrB[0][2][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 2]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][3][0][0][0]),
                      as_u32(tCrB[0][3][0][1][0]),
                      as_u32(tCrB[0][3][1][0][0]),
                      as_u32(tCrB[0][3][1][1][0]),
                      &b_smem[b_ldsm_base + Tiled_MMA_N * 3]);
  }

#pragma unroll 1
  for (; k_tiles_to_compute > 1; --k_tiles_to_compute) {
    n256_splitk::run_mma_tile_n256<K_PIPE_MAX - 2, true>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, stride_A, stride_B, tA_row, tA_col, tB_row, tB_col,
        warp_m_id, warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row,
        ldsmx4T_col);
  }

#pragma unroll
  for (; k_tiles_to_compute > -(K_PIPE_MAX - 1); --k_tiles_to_compute) {
    n256_splitk::run_mma_tile_n256<K_PIPE_MAX - 2, false>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, stride_A, stride_B, tA_row, tA_col, tB_row, tB_col,
        warp_m_id, warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row,
        ldsmx4T_col);
  }

  cp_async::wait_all();
  __syncthreads();

  float *sC = reinterpret_cast<float *>(shared_memory);
  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;
  constexpr int kSmemStrideC = 264;

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      n256_splitk::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][0][0], tCrC[m][n][0][0][1]);
      n256_splitk::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][1][0], tCrC[m][n][0][1][1]);
      n256_splitk::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][0][0], tCrC[m][n][1][0][1]);
      n256_splitk::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][1][0], tCrC[m][n][1][1][1]);
    }
  }

  __syncthreads();
  if (store_to_output) {
    store_gmem_f16_from_f32_recurrent<32, kCtaN, kStreamKThreads,
                                      kSmemStrideC>(
        reinterpret_cast<half *>(gOutput), sC, stride_B);
  } else {
    n256_splitk::store_gmem_f32_recurrent<32, kCtaN, kStreamKThreads,
                                          kSmemStrideC>(
        reinterpret_cast<float *>(gOutput), sC, kCtaN);
  }
  __syncthreads();
}

template <int kStages>
__device__ __forceinline__ int64_t process_one_segment(
    int64_t work, int64_t range_end, int tiled_shape_n,
    int iters_per_tile, int64_t work_per_block, const half *A,
    const half *B, float *partials,
    half *C,
    HgemmSharedStorage<shape_mnk_n256, kStages> *smem,
    char *shared_memory, int N, int K, int tile_id_offset,
    int dp_blocks, int max_peers_per_tile, bool store_to_output) {
  if (work < 0 || work >= range_end || tiled_shape_n <= 0 ||
      iters_per_tile <= 0 || work_per_block <= 0) {
    return sk::kInvalidWork;
  }

  int64_t tile_id_i64 = work / iters_per_tile;
  int64_t tile_work_begin = tile_id_i64 * iters_per_tile;
  int64_t segment_work_end =
      min(static_cast<int64_t>(range_end),
          tile_work_begin + static_cast<int64_t>(iters_per_tile));
  int64_t k_tile_begin = work - tile_work_begin;
  int64_t k_tile_end = segment_work_end - tile_work_begin;
  if (k_tile_begin >= k_tile_end) {
    return sk::kInvalidWork;
  }

  int local_tile_id = static_cast<int>(tile_id_i64);
  int output_tile_id = tile_id_offset + local_tile_id;
  int tile_m = output_tile_id / tiled_shape_n;
  int tile_n = output_tile_id % tiled_shape_n;
  char *gOutput = nullptr;
  if (!store_to_output) {
    int first_peer_block =
        static_cast<int>(tile_work_begin / work_per_block);
    int peer_slot = static_cast<int>(blockIdx.x) - dp_blocks - first_peer_block;
    if (peer_slot < 0 || peer_slot >= max_peers_per_tile) {
      return sk::kInvalidWork;
    }
    gOutput = reinterpret_cast<char *>(
        partials +
        (static_cast<int64_t>(local_tile_id) * max_peers_per_tile + peer_slot) *
            kStreamKTileElements);
  }
  int k_begin = static_cast<int>(k_tile_begin * kStreamKTileK);
  int k_end = static_cast<int>(k_tile_end * kStreamKTileK);
  const half *gA_base =
      A + tile_m * kStreamKTileM * K + k_begin;
  const half *gB_base =
      B + k_begin * N + tile_n * kStreamKTileN;
  if (store_to_output) {
    gOutput = reinterpret_cast<char *>(
        C + tile_m * kStreamKTileM * N + tile_n * kStreamKTileN);
  }
  compute_and_store_segment<kStages>(
      gA_base, gB_base, gOutput, smem, shared_memory, K, N,
      static_cast<int>((k_end - k_begin) / kMainloopTileK), store_to_output);
  return segment_work_end;
}

template <int kStages>
__global__ void sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_kernel(
    const half *A, const half *B, float *partials, half *C, int N, int K,
    StreamKParams params) {
  int tiled_shape_n = params.tiled_shape_n;
  int iters_per_tile = params.iters_per_tile;
  int dp_blocks = params.dp_blocks;
  int sk_tiles = params.sk_tiles;
  int sk_blocks = params.sk_blocks;
  int64_t sk_iters_per_block = params.sk_iters_per_block;
  int max_peers_per_tile = params.max_peers_per_tile;

  extern __shared__ char shared_memory[];
  auto *smem = reinterpret_cast<
      HgemmSharedStorage<shape_mnk_n256, kStages> *>(shared_memory);

  bool store_to_output = static_cast<int>(blockIdx.x) < dp_blocks;
  int sk_block = static_cast<int>(blockIdx.x) - dp_blocks;
  if (!store_to_output &&
      (sk_block < 0 || sk_block >= sk_blocks)) {
    return;
  }

  int64_t work_per_block = 0;
  int64_t range_begin = 0;
  int64_t range_end = 0;
  if (store_to_output) {
    work_per_block = iters_per_tile;
    range_begin = static_cast<int64_t>(blockIdx.x) * iters_per_tile;
    range_end = range_begin + iters_per_tile;
  } else {
    int64_t total_work = static_cast<int64_t>(sk_tiles) * iters_per_tile;
    work_per_block = sk_iters_per_block;
    range_begin = static_cast<int64_t>(sk_block) * work_per_block;
    range_end = min(total_work, range_begin + work_per_block);
  }
  if (range_begin >= range_end) {
    return;
  }

  for (int64_t work = range_begin; work < range_end;) {
    int64_t next_work = process_one_segment<kStages>(
        work, range_end, tiled_shape_n, iters_per_tile, work_per_block, A, B,
        partials, C, smem, shared_memory, N, K,
        store_to_output ? 0 : dp_blocks, dp_blocks, max_peers_per_tile,
        store_to_output);
    if (next_work <= work) {
      return;
    }
    work = next_work;
  }
}

__global__ void
sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_reduce_kernel(
    const float *__restrict__ partials, half *__restrict__ C, int N, int K,
    StreamKParams params) {
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerTile = kStreamKTileElements / kElementsPerVector;

  int tiled_shape_n = params.tiled_shape_n;
  int iters_per_tile = params.iters_per_tile;
  int sk_tiles = params.sk_tiles;
  int sk_blocks = params.sk_blocks;
  int64_t sk_iters_per_block = params.sk_iters_per_block;
  int max_peers_per_tile = params.max_peers_per_tile;
  int output_tile_offset = params.dp_blocks;
  int sk_tile_idx = static_cast<int>(blockIdx.x);
  if (sk_tiles <= 0 || tiled_shape_n <= 0 || iters_per_tile <= 0 ||
      sk_blocks <= 0 || max_peers_per_tile <= 0 ||
      sk_tile_idx >= sk_tiles) {
    return;
  }
  int peer_count = get_tile_peer_count(
      sk_tile_idx, sk_tiles, iters_per_tile, sk_blocks,
      sk_iters_per_block);
  if (peer_count <= 0 || peer_count > max_peers_per_tile) {
    return;
  }

  int output_tile_id = output_tile_offset + sk_tile_idx;
  int tile_m = output_tile_id / tiled_shape_n;
  int tile_n = output_tile_id % tiled_shape_n;
  int tid = threadIdx.x;
  int64_t partial_tile_base =
      static_cast<int64_t>(sk_tile_idx) * max_peers_per_tile *
      kStreamKTileElements;

  for (int vec = tid; vec < kVectorsPerTile; vec += blockDim.x) {
    int row = vec / (kStreamKTileN / kElementsPerVector);
    int col = (vec % (kStreamKTileN / kElementsPerVector)) *
              kElementsPerVector;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma unroll 1
    for (int peer = 0; peer < peer_count; ++peer) {
      const float *partial =
          partials + (partial_tile_base +
                      static_cast<int64_t>(peer) *
                          kStreamKTileElements);
      float4 value = *reinterpret_cast<const float4 *>(
          partial + row * kStreamKTileN + col);
      acc0 += value.x;
      acc1 += value.y;
      acc2 += value.z;
      acc3 += value.w;
    }

    long long output_index =
        (static_cast<long long>(tile_m) * kStreamKTileM + row) * N +
        static_cast<long long>(tile_n) * kStreamKTileN + col;
    half2 out01 = __floats2half2_rn(acc0, acc1);
    half2 out23 = __floats2half2_rn(acc2, acc3);
    half *out = C + output_index;
    *reinterpret_cast<half2 *>(out + 0) = out01;
    *reinterpret_cast<half2 *>(out + 2) = out23;
  }
}

inline cudaError_t configure_hgemm_128x256_streamk_fp32acc() {
  auto gemm_kernel =
      sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_kernel<kStreamKStages>;
  cudaError_t err = cudaFuncSetAttribute(
      gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kStreamKSharedStorageBytes128x256);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaFuncSetAttribute(gemm_kernel,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              100);
}

inline cudaError_t configure_hgemm_128x256_streamk_fp32acc(int block_swizzle) {
  (void)block_swizzle;
  return configure_hgemm_128x256_streamk_fp32acc();
}

inline void launch_hgemm_128x256_streamk_fp32acc_unchecked(
    const half *A, const half *B, float *partials, half *C, int N, int K,
    StreamKParams params,
    cudaStream_t stream = 0) {
  int total_blocks = params.dp_blocks + params.sk_blocks;
  dim3 block(kStreamKThreads);
  dim3 gemm_grid(static_cast<unsigned>(total_blocks));
  sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_kernel<kStreamKStages>
      <<<gemm_grid, block, kStreamKSharedStorageBytes128x256, stream>>>(
          A, B, partials, C, N, K, params);

  if (params.sk_tiles <= 0) {
    return;
  }
  dim3 reduce_grid(static_cast<unsigned>(params.sk_tiles));
  sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc_reduce_kernel
      <<<reduce_grid, block, 0, stream>>>(
          partials, C, N, K, params);
}

}  // namespace n256_streamk

} // namespace cuda_ops_core::detail::sm80_hgemm_128x256_streamk
