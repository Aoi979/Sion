#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

namespace cuda_ops_core::detail::sm80::ga10x {

constexpr int kCtaM = 128;
constexpr int kCtaN = 128;
constexpr int kCtaK = 8;
constexpr int kBlockSwizzle = 8;

constexpr int kWarpsM = 4;
constexpr int kWarpsN = 2;
constexpr int kWarpSize = 32;
constexpr int kThreads = kWarpsM * kWarpsN * kWarpSize;

constexpr int kWarpM = kCtaM / kWarpsM;
constexpr int kWarpN = kCtaN / kWarpsN;
constexpr int kWarpThreadsM = 4;
constexpr int kWarpThreadsN = 8;
constexpr int kThreadM = kWarpM / kWarpThreadsM;
constexpr int kThreadN = kWarpN / kWarpThreadsN;
constexpr int kLaneLayoutInterleave = 2;

constexpr int kSmemPadA = 4;
constexpr int kSmemPadB = 4;
constexpr int kSmemStrideA = kCtaM + kSmemPadA;
constexpr int kSmemStrideB = kCtaN + kSmemPadB;
constexpr int kSmemAStage = kCtaK * kSmemStrideA;
constexpr int kSmemBStage = kCtaK * kSmemStrideB;
constexpr int kStages = 5;
constexpr int kSharedStorageBytes =
    kStages * (kSmemAStage + kSmemBStage) * sizeof(float);

struct SharedStorage {
  struct Stage {
    float A[kSmemAStage];
    float B[kSmemBStage];
  } stage[kStages];
};

__device__ __forceinline__ unsigned smem_addr(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ float4 load_float4(const float *ptr) {
  return *reinterpret_cast<const float4 *>(ptr);
}

__device__ __forceinline__ void store_float4(float *ptr, float4 value) {
  *reinterpret_cast<float4 *>(ptr) = value;
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int Group>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(Group));
}

__device__ __forceinline__ void cp_async_4b(float *dst, const float *src) {
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" ::
                   "r"(smem_addr(dst)), "l"(src));
}

#define FFMA_ROW(C_ROW, A_COMP, B0, B1)                                      \
  tCrC[C_ROW][0] += (A_COMP) * (B0).x;                                       \
  tCrC[C_ROW][1] += (A_COMP) * (B0).y;                                       \
  tCrC[C_ROW][2] += (A_COMP) * (B0).z;                                       \
  tCrC[C_ROW][3] += (A_COMP) * (B0).w;                                       \
  tCrC[C_ROW][4] += (A_COMP) * (B1).x;                                       \
  tCrC[C_ROW][5] += (A_COMP) * (B1).y;                                       \
  tCrC[C_ROW][6] += (A_COMP) * (B1).z;                                       \
  tCrC[C_ROW][7] += (A_COMP) * (B1).w

#define STORE_4_ROWS(B_HALF)                                                   \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + lCRow * 4 + 0) * strideC +                    \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[0] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + lCRow * 4 + 1) * strideC +                    \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[1] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + lCRow * 4 + 2) * strideC +                    \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[2] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + lCRow * 4 + 3) * strideC +                    \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[3] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + kWarpM / 2 + lCRow * 4 + 0) * strideC +       \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[4] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + kWarpM / 2 + lCRow * 4 + 1) * strideC +       \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[5] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + kWarpM / 2 + lCRow * 4 + 2) * strideC +       \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[6] + (B_HALF) * 4));                                    \
  store_float4(                                                                \
      gC + (warpRow * kWarpM + kWarpM / 2 + lCRow * 4 + 3) * strideC +       \
                warpCol * kWarpN + (B_HALF) * (kWarpN / 2) + lCCol * 4,       \
      load_float4(tCrC[7] + (B_HALF) * 4))

template <int Group>
__device__ __forceinline__ void issue_tile_cp_async_group(
    float *stage_A_write, float *stage_B_write, const float *gA,
    const float *gB, int kStrideA, int kStrideB, int tA_row, int tA_col,
    int tB_row, int tB_col) {
  static_assert(Group >= 0 && Group < 4);
  constexpr int kARowOffset = Group * 32;
  constexpr int kBRowOffset = Group * 2;

  cp_async_4b(stage_A_write + tA_row + kARowOffset +
                 tA_col * kSmemStrideA,
             gA + (tA_row + kARowOffset) * kStrideA + tA_col);
  cp_async_4b(stage_B_write + (tB_row + kBRowOffset) * kSmemStrideB + tB_col,
             gB + (tB_row + kBRowOffset) * kStrideB + tB_col);
}

__device__ __forceinline__ void issue_tile_cp_async(
    float *stage_A_write, float *stage_B_write, const float *gA,
    const float *gB, int kStrideA, int kStrideB, int tA_row, int tA_col,
    int tB_row, int tB_col) {
  issue_tile_cp_async_group<0>(stage_A_write, stage_B_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
  issue_tile_cp_async_group<1>(stage_A_write, stage_B_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
  issue_tile_cp_async_group<2>(stage_A_write, stage_B_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
  issue_tile_cp_async_group<3>(stage_A_write, stage_B_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
}

__device__ __forceinline__ void issue_spread_cp_async_group(
    int kBlock, float *stage_A_write, float *stage_B_write,
    const float *gA_next, const float *gB_next, int kStrideA, int kStrideB,
    int tA_row, int tA_col, int tB_row, int tB_col) {
  if (kBlock == 0) {
    issue_tile_cp_async_group<0>(stage_A_write, stage_B_write, gA_next, gB_next,
                                 kStrideA, kStrideB, tA_row, tA_col, tB_row,
                                 tB_col);
  }
  if (kBlock == 2) {
    issue_tile_cp_async_group<1>(stage_A_write, stage_B_write, gA_next, gB_next,
                                 kStrideA, kStrideB, tA_row, tA_col, tB_row,
                                 tB_col);
  }
  if (kBlock == 4) {
    issue_tile_cp_async_group<2>(stage_A_write, stage_B_write, gA_next, gB_next,
                                 kStrideA, kStrideB, tA_row, tA_col, tB_row,
                                 tB_col);
  }
  if (kBlock == 6) {
    issue_tile_cp_async_group<3>(stage_A_write, stage_B_write, gA_next, gB_next,
                                 kStrideA, kStrideB, tA_row, tA_col, tB_row,
                                 tB_col);
  }
}

__global__ __launch_bounds__(kThreads, 2)
void sm80_GA10x_sgemm_f32_nn_m128n128k8_stage5_kernel(
    float *A, float *B, float *C, int N, int K) {
  const int strideA = K;
  const int strideB = N;
  const int strideC = N;
  const int tileNCount = N / kCtaN;
  const int tileM = blockIdx.x / kBlockSwizzle;
  const int tileN = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tileN >= tileNCount) {
    return;
  }

  const float *gA_base = A + tileM * kCtaM * strideA;
  const float *gB_base = B + tileN * kCtaN;
  float *gC = C + tileM * kCtaM * strideC + tileN * kCtaN;

  const int tid = threadIdx.x;
  const int warpId = tid / kWarpSize;
  const int warpRow = warpId / kWarpsN;
  const int warpCol = warpId % kWarpsN;
  const int laneId = tid % kWarpSize;
  const int lCRow = (laneId >> 4) * kLaneLayoutInterleave +
                   (laneId & (kLaneLayoutInterleave - 1));
  const int lCCol = (laneId / kLaneLayoutInterleave) & (kWarpThreadsN - 1);

  float tCrC[kThreadM][kThreadN] = {};
  float4 tCrA[kCtaK][kLaneLayoutInterleave];
  float4 tCrB[kCtaK][kLaneLayoutInterleave];

  extern __shared__ float smem_raw[];
  auto *stages = reinterpret_cast<SharedStorage *>(smem_raw);

  int tilesToIssue = K / kCtaK;
  int tilesToCompute = tilesToIssue;
  const int tARow = tid / kCtaK;
  const int tACol = tid % kCtaK;
  const int tBRow = tid / kCtaN;
  const int tBCol = tid % kCtaN;

#pragma unroll
  for (int pipe = 0; pipe < kStages - 1; ++pipe) {
    float *stageA = stages->stage[pipe].A;
    float *stageB = stages->stage[pipe].B;
    const float *gA = gA_base + pipe * kCtaK;
    const float *gB = gB_base + pipe * kCtaK * strideB;
    issue_tile_cp_async(stageA, stageB, gA, gB, strideA, strideB, tARow, tACol,
                        tBRow, tBCol);
    cp_async_commit_group();
    --tilesToIssue;
  }

  int smemPipeRead = 0;
  int smemPipeWrite = kStages - 1;
  float *stageAWrite = stages->stage[smemPipeWrite].A;
  float *stageBWrite = stages->stage[smemPipeWrite].B;
  const float *gANext = gA_base + (kStages - 1) * kCtaK;
  const float *gBNext = gB_base + (kStages - 1) * kCtaK * strideB;

  cp_async_wait_group<kStages - 2>();
  __syncthreads();

  float *stageARead = stages->stage[smemPipeRead].A;
  float *stageBRead = stages->stage[smemPipeRead].B;
  tCrA[0][0] = load_float4(stageARead + warpRow * kWarpM + lCRow * 4);
  tCrA[0][1] = load_float4(stageARead + warpRow * kWarpM + kWarpM / 2 +
                           lCRow * 4);
  tCrB[0][0] = load_float4(stageBRead + warpCol * kWarpN + lCCol * 4);
  tCrB[0][1] = load_float4(stageBRead + warpCol * kWarpN + kWarpN / 2 +
                           lCCol * 4);

  float *stageAP = stageARead;
  float *stageBP = stageBRead;
  while (tilesToCompute > 0) {
#pragma unroll
    for (int kBlock = 0; kBlock < kCtaK; ++kBlock) {
      if (kBlock == kCtaK - 1) {
        stageAP = stages->stage[smemPipeRead].A;
        stageBP = stages->stage[smemPipeRead].B;
        cp_async_wait_group<kStages - 2>();
        __syncthreads();
      }

      const int nextBlock = (kBlock + 1) % kCtaK;
      float *stageAK = stageAP + nextBlock * kSmemStrideA;
      float *stageBK = stageBP + nextBlock * kSmemStrideB;

      tCrA[nextBlock][0] =
          load_float4(stageAK + warpRow * kWarpM + lCRow * 4);
      tCrB[nextBlock][0] =
          load_float4(stageBK + warpCol * kWarpN + lCCol * 4);

      FFMA_ROW(0, tCrA[kBlock][0].x, tCrB[kBlock][0], tCrB[kBlock][1]);
      FFMA_ROW(1, tCrA[kBlock][0].y, tCrB[kBlock][0], tCrB[kBlock][1]);
      FFMA_ROW(2, tCrA[kBlock][0].z, tCrB[kBlock][0], tCrB[kBlock][1]);
      FFMA_ROW(3, tCrA[kBlock][0].w, tCrB[kBlock][0], tCrB[kBlock][1]);

      if (tilesToIssue > 0) {
        issue_spread_cp_async_group(
            kBlock, stageAWrite, stageBWrite, gANext, gBNext, strideA, strideB,
            tARow, tACol, tBRow, tBCol);
        if (kBlock == kCtaK - 2) {
          --tilesToIssue;
          gANext += kCtaK;
          gBNext += kCtaK * strideB;
        }
      }
      if (kBlock == kCtaK - 2) {
        cp_async_commit_group();
        smemPipeWrite = smemPipeRead;
        smemPipeRead = (smemPipeRead + 1) % kStages;
        stageAWrite = stages->stage[smemPipeWrite].A;
        stageBWrite = stages->stage[smemPipeWrite].B;
      }

      tCrA[nextBlock][1] =
          load_float4(stageAK + warpRow * kWarpM + kWarpM / 2 + lCRow * 4);
      tCrB[nextBlock][1] =
          load_float4(stageBK + warpCol * kWarpN + kWarpN / 2 + lCCol * 4);
      FFMA_ROW(4, tCrA[kBlock][1].x, tCrB[kBlock][0], tCrB[kBlock][1]);
      FFMA_ROW(5, tCrA[kBlock][1].y, tCrB[kBlock][0], tCrB[kBlock][1]);
      FFMA_ROW(6, tCrA[kBlock][1].z, tCrB[kBlock][0], tCrB[kBlock][1]);
      FFMA_ROW(7, tCrA[kBlock][1].w, tCrB[kBlock][0], tCrB[kBlock][1]);
    }
    --tilesToCompute;
  }

  STORE_4_ROWS(0);
  STORE_4_ROWS(1);
}

inline void launch(float *A, float *B, float *C, int M, int N, int K,
                   cudaStream_t stream = 0) {
  const int tileMCount = M / kCtaM;
  const int tileNCount = N / kCtaN;
  const dim3 block(kThreads);
  const dim3 grid(tileMCount * kBlockSwizzle,
                  (tileNCount + kBlockSwizzle - 1) / kBlockSwizzle);
  sm80_GA10x_sgemm_f32_nn_m128n128k8_stage5_kernel<<<
      grid, block, kSharedStorageBytes, stream>>>(A, B, C, N, K);
}

#undef STORE_4_ROWS
#undef FFMA_ROW

} // namespace cuda_ops_core::detail::sm80::ga10x
