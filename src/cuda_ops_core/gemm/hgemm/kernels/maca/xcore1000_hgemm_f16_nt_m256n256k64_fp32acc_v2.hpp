#pragma once

#include "../../detail/maca/mxc_builtins.hpp"

#include <stdint.h>

__device__ __forceinline__ uint16_t
xcore1000_hgemm_fp32_to_fp16_bits(float value) {
  __half converted = __float2half(value);
  __half_raw raw = converted;
  return raw.x;
}

__device__ __forceinline__ int xcore1000_hgemm_swizzled_k(int logical_row,
                                                          int logical_k) {
  return (((logical_k >> 3) ^ (logical_row & 7)) << 3);
}

#define XCORE1000_HGEMM_MMA(M, N, K_GROUP, K_IN_GROUP)                         \
  do {                                                                         \
    tCrC[(M)][(N)] = mxc::mma_16x16x16f16(tCrA[(M)][(K_GROUP)][(K_IN_GROUP)],  \
                                          tCrB[(N)][(K_GROUP)][(K_IN_GROUP)],  \
                                          tCrC[(M)][(N)]);                     \
  } while (0)

#define XCORE1000_HGEMM_LDG_A(K_TILE, STAGE)                                   \
  do {                                                                         \
    int const gmem_k_offset = (K_TILE) * kCtaK;                                \
    int const row = ldg_m_base + (STAGE) * kMmaAtomM;                          \
    mxc::ldg_b128_bsm<0, mxc::helper::all_lanes_mask, true, true, false,       \
                      true>(sA + row * kCtaK + ldg_smem_k_offset,              \
                            gA + row * gmem_a_stride + gmem_k_offset +         \
                                ldg_gmem_k_offset);                            \
  } while (0)

#define XCORE1000_HGEMM_LDG_B(K_TILE, STAGE)                                   \
  do {                                                                         \
    int const gmem_k_offset = (K_TILE) * kCtaK;                                \
    int const row = ldg_n_base + (STAGE) * (2 * kMmaAtomN);                    \
    mxc::ldg_b128_bsm<0, mxc::helper::all_lanes_mask, true, true, false,       \
                      true>(sB + row * kCtaK + ldg_smem_k_offset,              \
                            gB + row * gmem_b_stride + gmem_k_offset +         \
                                ldg_gmem_k_offset);                            \
  } while (0)

#define XCORE1000_HGEMM_LDS_A_FRAGMENT(STAGE, K_GROUP)                         \
  do {                                                                         \
    *reinterpret_cast<v4f32 *>(&tCrA[(STAGE)][(K_GROUP)]) =                    \
        *reinterpret_cast<v4f32 *>(                                            \
            sA + (lds_m_base + (STAGE) * kMmaAtomM) * kCtaK +                  \
            ((K_GROUP) == 0 ? lds_smem_k0 : lds_smem_k1));                     \
  } while (0)

#define XCORE1000_HGEMM_LDS_B_FRAGMENT(STAGE, N_IN_STAGE, K_GROUP)             \
  do {                                                                         \
    *reinterpret_cast<v4f32 *>(&tCrB[2 * (STAGE) + (N_IN_STAGE)][(K_GROUP)]) = \
        *reinterpret_cast<v4f32 *>(                                            \
            sB +                                                               \
            (((N_IN_STAGE) == 0 ? lds_n0_base : lds_n1_base) +                 \
             (STAGE) * 2 * kMmaAtomN) *                                        \
                kCtaK +                                                        \
            ((K_GROUP) == 0 ? lds_smem_k0 : lds_smem_k1));                     \
  } while (0)

__global__ void hgemm_tn_256x256x64_4stage_fp16(const void *A, const void *B,
                                                void *C, int M, int N, int K) {
  constexpr int kCtaM = 256;
  constexpr int kCtaN = 256;
  constexpr int kCtaK = 64;

  constexpr int kWaveSize = 64;
  constexpr int kThreads = 512;
  constexpr int kWaveTileM = 64;
  constexpr int kWaveTileN = 128;
  constexpr int kWaveNumsM = 4;

  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 16;
  constexpr int kMmaInstructionM = 4;
  constexpr int kMmaInstructionN = 8;

  constexpr int kLdgVectorSize = 8;
  constexpr int kLdgThreadsK = kCtaK / kLdgVectorSize; // 8
  constexpr int kLdgRowsMN = kThreads / kLdgThreadsK;  // 64
  constexpr int kStage = kCtaM / kLdgRowsMN;           // 4
  constexpr int kSmemSize =
      (kCtaM * kCtaK + kCtaN * kCtaK) * sizeof(half); // 64 KiB
  static_assert(kStage == 4, "mainloop schedule requires four row stages");
  static_assert(kLdgVectorSize == 8 && kLdgThreadsK == 8,
                "K-chunk XOR requires eight aligned 16-byte chunks per row");

  int const tid = threadIdx.x;
  int const lane_id = tid % kWaveSize;
  int const wave_id = tid / kWaveSize;
  int const wave_id_m = wave_id % kWaveNumsM;
  int const wave_id_n = wave_id / kWaveNumsM;
  int const tile_m = blockIdx.x;
  int const tile_n = blockIdx.y;

  if ((tile_m + 1) * kCtaM > M || (tile_n + 1) * kCtaN > N || K < kCtaK ||
      K % kCtaK != 0) {
    return;
  }

  __shared__ uint8_t smem[kSmemSize];
  half *sA = reinterpret_cast<half *>(smem);
  half *sB = sA + kCtaM * kCtaK;

  int const gmem_a_stride = K;
  int const gmem_b_stride = K;
  half *gA = reinterpret_cast<half *>(const_cast<void *>(A)) +
             tile_m * kCtaM * gmem_a_stride;
  half *gB = reinterpret_cast<half *>(const_cast<void *>(B)) +
             tile_n * kCtaN * gmem_b_stride;

  v4f16 tCrA[kMmaInstructionM][2][2];
  v4f16 tCrB[kMmaInstructionN][2][2];
  v4f32 tCrC[kMmaInstructionM][kMmaInstructionN] = {};

  int const ldg_smem_k_offset = (tid % kLdgThreadsK) * kLdgVectorSize;
  int const ldg_gmem_k_offset =
      xcore1000_hgemm_swizzled_k(tid / kLdgThreadsK, ldg_smem_k_offset);

  int const ldg_m_row = tid / kLdgThreadsK;
  int const ldg_m_group = ldg_m_row / kMmaAtomM;
  int const ldg_m_base = ldg_m_group * kWaveTileM + ldg_m_row % kMmaAtomM;

  int const ldg_n_row = tid / kLdgThreadsK;
  int const ldg_n_group = ldg_n_row / (2 * kMmaAtomN);
  int const ldg_n_base = ldg_n_group * kWaveTileN + ldg_n_row % (2 * kMmaAtomN);

  int const mma_mn = tid % kMmaAtomM;
  int const lds_logical_k0 = (lane_id / kMmaAtomM) * kLdgVectorSize;
  int const lds_smem_k0 = xcore1000_hgemm_swizzled_k(mma_mn, lds_logical_k0);
  int const lds_smem_k1 =
      xcore1000_hgemm_swizzled_k(mma_mn, lds_logical_k0 + 32);
  int const lds_m_base = wave_id_m * kWaveTileM + mma_mn;
  int const lds_n0_base = wave_id_n * kWaveTileN + mma_mn;
  int const lds_n1_base = lds_n0_base + kMmaAtomN;

  XCORE1000_HGEMM_LDG_A(0, 0);
  XCORE1000_HGEMM_LDG_B(0, 0);
  XCORE1000_HGEMM_LDG_A(0, 1);
  XCORE1000_HGEMM_LDG_B(0, 1);
  XCORE1000_HGEMM_LDG_A(0, 2);
  XCORE1000_HGEMM_LDG_B(0, 2);
  XCORE1000_HGEMM_LDG_A(0, 3);
  XCORE1000_HGEMM_LDG_B(0, 3);

  mxc::arrive_gvmcnt<6>();
  mxc::barrier_inst();
  XCORE1000_HGEMM_LDS_A_FRAGMENT(0, 0);
  XCORE1000_HGEMM_LDS_A_FRAGMENT(0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 0, 0);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 1, 0);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 1, 1);

  mxc::arrive_gvmcnt<4>();
  mxc::barrier_inst();
  XCORE1000_HGEMM_LDS_A_FRAGMENT(1, 0);
  XCORE1000_HGEMM_LDS_A_FRAGMENT(1, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 0, 0);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 1, 0);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 1, 1);

  int const k_tile_num = K / kCtaK;

  // Stable-state invariant at loop entry:
  //   - current stage0/1 are in registers;
  //   - current stage2/3 are the four outstanding gmem->smem requests.
  for (int k_tile = 0; k_tile < k_tile_num - 1; ++k_tile) {
    int const next_tile = k_tile + 1;

    XCORE1000_HGEMM_MMA(0, 0, 0, 0);
    XCORE1000_HGEMM_LDG_A(next_tile, 0);
    XCORE1000_HGEMM_MMA(0, 0, 0, 1);
    XCORE1000_HGEMM_MMA(0, 0, 1, 0);
    XCORE1000_HGEMM_MMA(0, 0, 1, 1);
    XCORE1000_HGEMM_MMA(1, 0, 0, 0);
    XCORE1000_HGEMM_MMA(1, 0, 0, 1);
    XCORE1000_HGEMM_MMA(1, 0, 1, 0);
    XCORE1000_HGEMM_MMA(1, 0, 1, 1);
    XCORE1000_HGEMM_MMA(0, 1, 0, 0);
    XCORE1000_HGEMM_MMA(0, 1, 0, 1);
    XCORE1000_HGEMM_MMA(0, 1, 1, 0);
    XCORE1000_HGEMM_MMA(0, 1, 1, 1);
    XCORE1000_HGEMM_MMA(1, 1, 0, 0);
    XCORE1000_HGEMM_MMA(1, 1, 0, 1);
    XCORE1000_HGEMM_MMA(1, 1, 1, 0);

    mxc::arrive_gvmcnt<3>();
    mxc::barrier_inst();
    XCORE1000_HGEMM_MMA(1, 1, 1, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(2, 0);
    XCORE1000_HGEMM_MMA(0, 2, 0, 0);
    XCORE1000_HGEMM_LDG_B(next_tile, 0);
    XCORE1000_HGEMM_MMA(0, 2, 0, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(2, 1);
    XCORE1000_HGEMM_MMA(0, 2, 1, 0);
    XCORE1000_HGEMM_MMA(0, 2, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 0, 0);
    XCORE1000_HGEMM_MMA(1, 2, 0, 0);
    XCORE1000_HGEMM_MMA(1, 2, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 0, 1);
    XCORE1000_HGEMM_MMA(1, 2, 1, 0);
    XCORE1000_HGEMM_MMA(1, 2, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 1, 0);
    XCORE1000_HGEMM_MMA(0, 3, 0, 0);
    XCORE1000_HGEMM_MMA(0, 3, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 1, 1);
    XCORE1000_HGEMM_MMA(0, 3, 1, 0);
    XCORE1000_HGEMM_MMA(0, 3, 1, 1);
    XCORE1000_HGEMM_MMA(1, 3, 0, 0);
    XCORE1000_HGEMM_MMA(1, 3, 0, 1);
    XCORE1000_HGEMM_MMA(1, 3, 1, 0);

    XCORE1000_HGEMM_MMA(1, 3, 1, 1);

    XCORE1000_HGEMM_MMA(2, 0, 0, 0);
    XCORE1000_HGEMM_LDG_A(next_tile, 1);
    XCORE1000_HGEMM_MMA(2, 0, 0, 1);
    XCORE1000_HGEMM_MMA(2, 0, 1, 0);
    XCORE1000_HGEMM_MMA(2, 0, 1, 1);
    XCORE1000_HGEMM_MMA(2, 1, 0, 0);
    XCORE1000_HGEMM_MMA(2, 1, 0, 1);
    XCORE1000_HGEMM_MMA(2, 1, 1, 0);
    XCORE1000_HGEMM_MMA(2, 1, 1, 1);
    XCORE1000_HGEMM_MMA(2, 4, 0, 0);
    XCORE1000_HGEMM_MMA(2, 4, 0, 1);
    XCORE1000_HGEMM_MMA(2, 4, 1, 0);
    XCORE1000_HGEMM_MMA(2, 4, 1, 1);
    XCORE1000_HGEMM_MMA(0, 4, 0, 0);
    XCORE1000_HGEMM_MMA(0, 4, 0, 1);
    XCORE1000_HGEMM_MMA(0, 4, 1, 0);

    mxc::arrive_gvmcnt<3>();
    mxc::barrier_inst();
    XCORE1000_HGEMM_MMA(0, 4, 1, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(3, 0);
    XCORE1000_HGEMM_MMA(2, 5, 0, 0);
    XCORE1000_HGEMM_LDG_B(next_tile, 1);
    XCORE1000_HGEMM_MMA(2, 5, 0, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(3, 1);
    XCORE1000_HGEMM_MMA(2, 5, 1, 0);
    XCORE1000_HGEMM_MMA(2, 5, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 0, 0);
    XCORE1000_HGEMM_MMA(0, 5, 0, 0);
    XCORE1000_HGEMM_MMA(0, 5, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 0, 1);
    XCORE1000_HGEMM_MMA(0, 5, 1, 0);
    XCORE1000_HGEMM_MMA(0, 5, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 1, 0);
    XCORE1000_HGEMM_MMA(1, 4, 0, 0);
    XCORE1000_HGEMM_MMA(1, 4, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 1, 1);
    XCORE1000_HGEMM_MMA(1, 4, 1, 0);
    XCORE1000_HGEMM_MMA(1, 4, 1, 1);
    XCORE1000_HGEMM_MMA(1, 5, 0, 0);
    XCORE1000_HGEMM_MMA(1, 5, 0, 1);
    XCORE1000_HGEMM_MMA(1, 5, 1, 0);

    XCORE1000_HGEMM_MMA(1, 5, 1, 1);

    XCORE1000_HGEMM_MMA(0, 6, 0, 0);
    XCORE1000_HGEMM_LDG_A(next_tile, 2);
    XCORE1000_HGEMM_MMA(0, 6, 0, 1);
    XCORE1000_HGEMM_MMA(0, 6, 1, 0);
    XCORE1000_HGEMM_MMA(0, 6, 1, 1);
    XCORE1000_HGEMM_MMA(0, 7, 0, 0);
    XCORE1000_HGEMM_MMA(0, 7, 0, 1);
    XCORE1000_HGEMM_MMA(0, 7, 1, 0);
    XCORE1000_HGEMM_MMA(0, 7, 1, 1);
    XCORE1000_HGEMM_MMA(3, 0, 0, 0);
    XCORE1000_HGEMM_MMA(3, 0, 0, 1);
    XCORE1000_HGEMM_MMA(3, 0, 1, 0);
    XCORE1000_HGEMM_MMA(3, 0, 1, 1);
    XCORE1000_HGEMM_MMA(3, 1, 0, 0);
    XCORE1000_HGEMM_MMA(3, 1, 0, 1);
    XCORE1000_HGEMM_MMA(3, 1, 1, 0);

    mxc::arrive_gvmcnt<3>();
    mxc::barrier_inst();
    XCORE1000_HGEMM_MMA(3, 1, 1, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(0, 0);

    XCORE1000_HGEMM_MMA(3, 4, 0, 0);
    XCORE1000_HGEMM_LDG_B(next_tile, 2);
    XCORE1000_HGEMM_MMA(3, 4, 0, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(0, 1);
    XCORE1000_HGEMM_MMA(3, 4, 1, 0);
    XCORE1000_HGEMM_MMA(3, 4, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 0, 0);
    XCORE1000_HGEMM_MMA(3, 5, 0, 0);
    XCORE1000_HGEMM_MMA(3, 5, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 0, 1);
    XCORE1000_HGEMM_MMA(3, 5, 1, 0);
    XCORE1000_HGEMM_MMA(3, 5, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 1, 0);
    XCORE1000_HGEMM_MMA(1, 6, 0, 0);
    XCORE1000_HGEMM_MMA(1, 6, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(0, 1, 1);
    XCORE1000_HGEMM_MMA(1, 6, 1, 0);
    XCORE1000_HGEMM_MMA(1, 6, 1, 1);
    XCORE1000_HGEMM_MMA(1, 7, 0, 0);
    XCORE1000_HGEMM_MMA(1, 7, 0, 1);
    XCORE1000_HGEMM_MMA(1, 7, 1, 0);
    XCORE1000_HGEMM_MMA(1, 7, 1, 1);

    XCORE1000_HGEMM_MMA(2, 2, 0, 0);
    XCORE1000_HGEMM_LDG_A(next_tile, 3);
    XCORE1000_HGEMM_MMA(2, 2, 0, 1);
    XCORE1000_HGEMM_MMA(2, 2, 1, 0);
    XCORE1000_HGEMM_MMA(2, 2, 1, 1);
    XCORE1000_HGEMM_MMA(3, 2, 0, 0);
    XCORE1000_HGEMM_MMA(3, 2, 0, 1);
    XCORE1000_HGEMM_MMA(3, 2, 1, 0);
    XCORE1000_HGEMM_MMA(3, 2, 1, 1);
    XCORE1000_HGEMM_MMA(2, 3, 0, 0);
    XCORE1000_HGEMM_MMA(2, 3, 0, 1);
    XCORE1000_HGEMM_MMA(2, 3, 1, 0);
    XCORE1000_HGEMM_MMA(2, 3, 1, 1);
    XCORE1000_HGEMM_MMA(3, 3, 0, 0);
    XCORE1000_HGEMM_MMA(3, 3, 0, 1);
    XCORE1000_HGEMM_MMA(3, 3, 1, 0);

    mxc::arrive_gvmcnt<3>();
    mxc::barrier_inst();
    XCORE1000_HGEMM_MMA(3, 3, 1, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(1, 0);
    XCORE1000_HGEMM_MMA(2, 6, 0, 0);
    XCORE1000_HGEMM_LDG_B(next_tile, 3);
    XCORE1000_HGEMM_MMA(2, 6, 0, 1);
    XCORE1000_HGEMM_LDS_A_FRAGMENT(1, 1);
    XCORE1000_HGEMM_MMA(2, 6, 1, 0);
    XCORE1000_HGEMM_MMA(2, 6, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 0, 0);
    XCORE1000_HGEMM_MMA(3, 6, 0, 0);
    XCORE1000_HGEMM_MMA(3, 6, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 0, 1);
    XCORE1000_HGEMM_MMA(3, 6, 1, 0);
    XCORE1000_HGEMM_MMA(3, 6, 1, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 1, 0);
    XCORE1000_HGEMM_MMA(2, 7, 0, 0);
    XCORE1000_HGEMM_MMA(2, 7, 0, 1);
    XCORE1000_HGEMM_LDS_B_FRAGMENT(1, 1, 1);
    XCORE1000_HGEMM_MMA(2, 7, 1, 0);
    XCORE1000_HGEMM_MMA(2, 7, 1, 1);
    XCORE1000_HGEMM_MMA(3, 7, 0, 0);
    XCORE1000_HGEMM_MMA(3, 7, 0, 1);
    XCORE1000_HGEMM_MMA(3, 7, 1, 0);
    XCORE1000_HGEMM_MMA(3, 7, 1, 1);
  }

  // Drain
  XCORE1000_HGEMM_MMA(0, 0, 0, 0);
  XCORE1000_HGEMM_MMA(0, 0, 0, 1);
  XCORE1000_HGEMM_MMA(0, 0, 1, 0);
  XCORE1000_HGEMM_MMA(0, 0, 1, 1);
  XCORE1000_HGEMM_MMA(1, 0, 0, 0);
  XCORE1000_HGEMM_MMA(1, 0, 0, 1);
  XCORE1000_HGEMM_MMA(1, 0, 1, 0);
  XCORE1000_HGEMM_MMA(1, 0, 1, 1);
  XCORE1000_HGEMM_MMA(0, 1, 0, 0);
  XCORE1000_HGEMM_MMA(0, 1, 0, 1);
  XCORE1000_HGEMM_MMA(0, 1, 1, 0);
  XCORE1000_HGEMM_MMA(0, 1, 1, 1);
  XCORE1000_HGEMM_MMA(1, 1, 0, 0);
  XCORE1000_HGEMM_MMA(1, 1, 0, 1);
  XCORE1000_HGEMM_MMA(1, 1, 1, 0);
  mxc::arrive_gvmcnt<2>();
  mxc::barrier_inst();
  XCORE1000_HGEMM_MMA(1, 1, 1, 1);
  XCORE1000_HGEMM_LDS_A_FRAGMENT(2, 0);
  XCORE1000_HGEMM_MMA(0, 2, 0, 0);
  XCORE1000_HGEMM_MMA(0, 2, 0, 1);
  XCORE1000_HGEMM_LDS_A_FRAGMENT(2, 1);
  XCORE1000_HGEMM_MMA(0, 2, 1, 0);
  XCORE1000_HGEMM_MMA(0, 2, 1, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 0, 0);
  XCORE1000_HGEMM_MMA(1, 2, 0, 0);
  XCORE1000_HGEMM_MMA(1, 2, 0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 0, 1);
  XCORE1000_HGEMM_MMA(1, 2, 1, 0);
  XCORE1000_HGEMM_MMA(1, 2, 1, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 1, 0);
  XCORE1000_HGEMM_MMA(0, 3, 0, 0);
  XCORE1000_HGEMM_MMA(0, 3, 0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(2, 1, 1);
  XCORE1000_HGEMM_MMA(0, 3, 1, 0);
  XCORE1000_HGEMM_MMA(0, 3, 1, 1);
  XCORE1000_HGEMM_MMA(1, 3, 0, 0);
  XCORE1000_HGEMM_MMA(1, 3, 0, 1);
  XCORE1000_HGEMM_MMA(1, 3, 1, 0);
  XCORE1000_HGEMM_MMA(1, 3, 1, 1);
  XCORE1000_HGEMM_MMA(2, 0, 0, 0);
  XCORE1000_HGEMM_MMA(2, 0, 0, 1);
  XCORE1000_HGEMM_MMA(2, 0, 1, 0);
  XCORE1000_HGEMM_MMA(2, 0, 1, 1);
  XCORE1000_HGEMM_MMA(2, 1, 0, 0);
  XCORE1000_HGEMM_MMA(2, 1, 0, 1);
  XCORE1000_HGEMM_MMA(2, 1, 1, 0);
  XCORE1000_HGEMM_MMA(2, 1, 1, 1);
  XCORE1000_HGEMM_MMA(2, 4, 0, 0);
  XCORE1000_HGEMM_MMA(2, 4, 0, 1);
  XCORE1000_HGEMM_MMA(2, 4, 1, 0);
  XCORE1000_HGEMM_MMA(2, 4, 1, 1);
  XCORE1000_HGEMM_MMA(0, 4, 0, 0);
  XCORE1000_HGEMM_MMA(0, 4, 0, 1);
  XCORE1000_HGEMM_MMA(0, 4, 1, 0);
  mxc::arrive_gvmcnt<0>();
  mxc::barrier_inst();
  XCORE1000_HGEMM_MMA(0, 4, 1, 1);
  XCORE1000_HGEMM_LDS_A_FRAGMENT(3, 0);
  XCORE1000_HGEMM_MMA(2, 5, 0, 0);
  XCORE1000_HGEMM_MMA(2, 5, 0, 1);
  XCORE1000_HGEMM_LDS_A_FRAGMENT(3, 1);
  XCORE1000_HGEMM_MMA(2, 5, 1, 0);
  XCORE1000_HGEMM_MMA(2, 5, 1, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 0, 0);
  XCORE1000_HGEMM_MMA(0, 5, 0, 0);
  XCORE1000_HGEMM_MMA(0, 5, 0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 0, 1);
  XCORE1000_HGEMM_MMA(0, 5, 1, 0);
  XCORE1000_HGEMM_MMA(0, 5, 1, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 1, 0);
  XCORE1000_HGEMM_MMA(1, 4, 0, 0);
  XCORE1000_HGEMM_MMA(1, 4, 0, 1);
  XCORE1000_HGEMM_LDS_B_FRAGMENT(3, 1, 1);
  XCORE1000_HGEMM_MMA(1, 4, 1, 0);
  XCORE1000_HGEMM_MMA(1, 4, 1, 1);
  XCORE1000_HGEMM_MMA(1, 5, 0, 0);
  XCORE1000_HGEMM_MMA(1, 5, 0, 1);
  XCORE1000_HGEMM_MMA(1, 5, 1, 0);
  XCORE1000_HGEMM_MMA(1, 5, 1, 1);
  XCORE1000_HGEMM_MMA(3, 0, 0, 0);
  XCORE1000_HGEMM_MMA(3, 0, 0, 1);
  XCORE1000_HGEMM_MMA(3, 0, 1, 0);
  XCORE1000_HGEMM_MMA(3, 0, 1, 1);
  XCORE1000_HGEMM_MMA(3, 1, 0, 0);
  XCORE1000_HGEMM_MMA(3, 1, 0, 1);
  XCORE1000_HGEMM_MMA(3, 1, 1, 0);
  XCORE1000_HGEMM_MMA(3, 1, 1, 1);
  XCORE1000_HGEMM_MMA(3, 4, 0, 0);
  XCORE1000_HGEMM_MMA(3, 4, 0, 1);
  XCORE1000_HGEMM_MMA(3, 4, 1, 0);
  XCORE1000_HGEMM_MMA(3, 4, 1, 1);
  XCORE1000_HGEMM_MMA(3, 5, 0, 0);
  XCORE1000_HGEMM_MMA(3, 5, 0, 1);
  XCORE1000_HGEMM_MMA(3, 5, 1, 0);
  XCORE1000_HGEMM_MMA(3, 5, 1, 1);
  XCORE1000_HGEMM_MMA(0, 6, 0, 0);
  XCORE1000_HGEMM_MMA(0, 6, 0, 1);
  XCORE1000_HGEMM_MMA(0, 6, 1, 0);
  XCORE1000_HGEMM_MMA(0, 6, 1, 1);
  XCORE1000_HGEMM_MMA(0, 7, 0, 0);
  XCORE1000_HGEMM_MMA(0, 7, 0, 1);
  XCORE1000_HGEMM_MMA(0, 7, 1, 0);
  XCORE1000_HGEMM_MMA(0, 7, 1, 1);
  XCORE1000_HGEMM_MMA(1, 6, 0, 0);
  XCORE1000_HGEMM_MMA(1, 6, 0, 1);
  XCORE1000_HGEMM_MMA(1, 6, 1, 0);
  XCORE1000_HGEMM_MMA(1, 6, 1, 1);
  XCORE1000_HGEMM_MMA(1, 7, 0, 0);
  XCORE1000_HGEMM_MMA(1, 7, 0, 1);
  XCORE1000_HGEMM_MMA(1, 7, 1, 0);
  XCORE1000_HGEMM_MMA(1, 7, 1, 1);
  XCORE1000_HGEMM_MMA(2, 2, 0, 0);
  XCORE1000_HGEMM_MMA(2, 2, 0, 1);
  XCORE1000_HGEMM_MMA(2, 2, 1, 0);
  XCORE1000_HGEMM_MMA(2, 2, 1, 1);
  XCORE1000_HGEMM_MMA(3, 2, 0, 0);
  XCORE1000_HGEMM_MMA(3, 2, 0, 1);
  XCORE1000_HGEMM_MMA(3, 2, 1, 0);
  XCORE1000_HGEMM_MMA(3, 2, 1, 1);
  XCORE1000_HGEMM_MMA(2, 3, 0, 0);
  XCORE1000_HGEMM_MMA(2, 3, 0, 1);
  XCORE1000_HGEMM_MMA(2, 3, 1, 0);
  XCORE1000_HGEMM_MMA(2, 3, 1, 1);
  XCORE1000_HGEMM_MMA(3, 3, 0, 0);
  XCORE1000_HGEMM_MMA(3, 3, 0, 1);
  XCORE1000_HGEMM_MMA(3, 3, 1, 0);
  XCORE1000_HGEMM_MMA(3, 3, 1, 1);
  XCORE1000_HGEMM_MMA(2, 6, 0, 0);
  XCORE1000_HGEMM_MMA(2, 6, 0, 1);
  XCORE1000_HGEMM_MMA(2, 6, 1, 0);
  XCORE1000_HGEMM_MMA(2, 6, 1, 1);
  XCORE1000_HGEMM_MMA(3, 6, 0, 0);
  XCORE1000_HGEMM_MMA(3, 6, 0, 1);
  XCORE1000_HGEMM_MMA(3, 6, 1, 0);
  XCORE1000_HGEMM_MMA(3, 6, 1, 1);
  XCORE1000_HGEMM_MMA(2, 7, 0, 0);
  XCORE1000_HGEMM_MMA(2, 7, 0, 1);
  XCORE1000_HGEMM_MMA(2, 7, 1, 0);
  XCORE1000_HGEMM_MMA(2, 7, 1, 1);
  XCORE1000_HGEMM_MMA(3, 7, 0, 0);
  XCORE1000_HGEMM_MMA(3, 7, 0, 1);
  XCORE1000_HGEMM_MMA(3, 7, 1, 0);
  XCORE1000_HGEMM_MMA(3, 7, 1, 1);

  // Epilogue
  uint16_t *gC =
      reinterpret_cast<uint16_t *>(C) + tile_m * kCtaM * N + tile_n * kCtaN;
  int const output_row_base =
      wave_id_m * kWaveTileM + (lane_id / kMmaAtomM) * 4;
  int const output_col_base = wave_id_n * kWaveTileN + mma_mn;

#pragma unroll
  for (int mma_m = 0; mma_m < kMmaInstructionM; ++mma_m) {
#pragma unroll
    for (int mma_n = 0; mma_n < kMmaInstructionN; ++mma_n) {
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        int const row = output_row_base + mma_m * kMmaAtomM + element;
        int const col = output_col_base + mma_n * kMmaAtomN;
        float const value =
            reinterpret_cast<float const *>(&tCrC[mma_m][mma_n])[element];
        gC[row * N + col] = xcore1000_hgemm_fp32_to_fp16_bits(value);
      }
    }
  }
}

#undef XCORE1000_HGEMM_MMA
#undef XCORE1000_HGEMM_LDG_A
#undef XCORE1000_HGEMM_LDG_B
#undef XCORE1000_HGEMM_LDS_A_FRAGMENT
#undef XCORE1000_HGEMM_LDS_B_FRAGMENT
