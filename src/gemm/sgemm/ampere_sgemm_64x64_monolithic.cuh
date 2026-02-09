#include <cstdint>
#include <utils/macro.h>

// General version, therefore sacrificing vectorized memory access optimization
__global__ void ampere_sgemm_64x64(int M, int N, int K, float alpha,
                                   float const* __restrict__ A, float const* __restrict__ B, float beta,
                                   float * __restrict__ C) {
  constexpr uint32_t WARP_SIZE = 32;

  constexpr uint32_t kBM = 64;
  constexpr uint32_t kBN = 64;
  constexpr uint32_t kBK = 8;
  constexpr uint32_t kWM = 32;
  constexpr uint32_t kWN = 64;
  constexpr uint32_t kTM = 4;
  constexpr uint32_t kTN = 4;
  constexpr uint32_t kWM_ITER = 2;
  constexpr uint32_t kWN_ITER = 2;
  constexpr uint32_t kStages = 2;

  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kWarps = kThreads / WARP_SIZE;

  constexpr uint32_t load_smem_A_row_stride = kThreads / kBK;
  constexpr uint32_t load_smem_B_row_stride = kThreads / kBN;
  constexpr uint32_t load_a_iters = kBM / load_smem_A_row_stride;
  constexpr uint32_t load_b_iters = kBK / load_smem_B_row_stride;

  __shared__ float aSmemTile[kStages][kBK * kBM];
  __shared__ float bSmemTile[kStages][kBK * kBN];

  float aFragment[kStages][kWM_ITER * kTM] = {0.0};
  float bFragment[kStages][kWN_ITER * kTN] = {0.0};
  float result[kWM_ITER * kWN_ITER * kTM * kTN] = {0.0};

  float ldg_a[load_a_iters];
  float ldg_b[load_b_iters];

  uint32_t tid = threadIdx.x;
  uint32_t warp_id = threadIdx.x / WARP_SIZE;
  uint32_t lane_id = threadIdx.x % WARP_SIZE;
  constexpr uint32_t warps_per_row = kBN / kWN;
  uint32_t warp_row = warp_id / warps_per_row;
  constexpr uint32_t warp_col = 0;
  constexpr uint32_t warpMMATileM = kWM / kWM_ITER;
  constexpr uint32_t warpMMATileN = kWN / kWN_ITER;
  uint32_t thread_row = lane_id / (warpMMATileN / kTN);
  uint32_t thread_col = lane_id % (warpMMATileN / kTN);
  uint32_t k_iter = (K + kBK - 1) / kBK;

  A += blockIdx.y * kBM * K;
  B += blockIdx.x * kBN;
  C += blockIdx.y * kBM * N + blockIdx.x * kBN;

  constexpr uint32_t load_a_row_thread_num = kBK;
  constexpr uint32_t load_b_row_thread_num = kBN;

  uint32_t load_smem_a_row = tid / load_a_row_thread_num;
  uint32_t load_smem_a_col = tid % load_a_row_thread_num;

  uint32_t load_smem_b_row = tid / load_b_row_thread_num;
  uint32_t load_smem_b_col = tid % load_b_row_thread_num;

  // prologue
  {
    // gmem2smem <- gmem2reg2smem
#pragma unroll
    for (uint32_t i = 0; i < load_a_iters; i++) {
      uint32_t gk = 0 * kBK + load_smem_a_col;
      uint32_t gm =
          blockIdx.y * kBM + i * load_smem_A_row_stride + load_smem_a_row;
      uint32_t load_smem_a =
          load_smem_a_col * kBM + load_smem_a_row + i * load_smem_A_row_stride;
      if (gk >= K || gm >= M) {
        ldg_a[i] = 0.0f;
      } else {
        uint32_t load_gmem_a =
            (i * load_smem_A_row_stride + load_smem_a_row) * K + gk;
        ldg_a[i] = A[load_gmem_a];
      }
      aSmemTile[0][load_smem_a] = ldg_a[i];
    }

#pragma unroll
    for (uint32_t i = 0; i < load_b_iters; i++) {
      uint32_t gk = 0 * kBK + (i * load_smem_B_row_stride + load_smem_b_row);
      uint32_t gn = blockIdx.x * kBN + load_smem_b_col;
      uint32_t load_smem_b =
          (load_smem_b_row + i * load_smem_B_row_stride) * kBN +
          load_smem_b_col;
      if (gk >= K || gn >= N) {
        ldg_b[i] = 0.0f;
      } else {
        uint32_t load_gmem_b = gk * N + load_smem_b_col;
        ldg_b[i] = B[load_gmem_b];
      }
      bSmemTile[0][load_smem_b] = ldg_b[i];
    }

    __syncthreads();

    float *aSmemWarpTile = &aSmemTile[0][warp_row * kWM];
    float *bSmemWarpTile = &bSmemTile[0][warp_col * kWN];

    // smem2reg
#pragma unroll
    for (int a = 0; a < kWM_ITER; a++) {
      FETCH_FLOAT4(aFragment[0][a * kTM]) = FETCH_FLOAT4(
          aSmemWarpTile[thread_row * kTM + 0 * kBM + a * warpMMATileM]);
    }
#pragma unroll
    for (int b = 0; b < kWN_ITER; b++) {
      FETCH_FLOAT4(bFragment[0][b * kTN]) = FETCH_FLOAT4(
          bSmemWarpTile[thread_col * kTN + 0 * kBN + b * warpMMATileN]);
    }
  }
  // mainloop
  {
    uint32_t write_stage_idx = 1;

    for (uint32_t k = 1; k < k_iter; k++) {
#pragma unroll
      for (uint32_t i = 0; i < load_a_iters; i++) {
        uint32_t gk = k * kBK + load_smem_a_col;
        uint32_t gm =
            blockIdx.y * kBM + i * load_smem_A_row_stride + load_smem_a_row;
        if (gk >= K || gm >= M) {
          ldg_a[i] = 0.0f;
        } else {
          uint32_t load_gmem_a =
              (i * load_smem_A_row_stride + load_smem_a_row) * K + gk;
          ldg_a[i] = A[load_gmem_a];
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < load_b_iters; i++) {
        uint32_t gk = k * kBK + (i * load_smem_B_row_stride + load_smem_b_row);
        uint32_t gn = blockIdx.x * kBN + load_smem_b_col;
        if (gk >= K || gn >= N) {
          ldg_b[i] = 0.0f;
        } else {
          uint32_t load_gmem_b = gk * N + load_smem_b_col;
          ldg_b[i] = B[load_gmem_b];
        }
      }

      uint32_t load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
      for (uint32_t dot_product_idx = 1; dot_product_idx < kBK;
           ++dot_product_idx) {
        float *aSmemWarpTile = &aSmemTile[load_stage_idx][warp_row * kWM];
        float *bSmemWarpTile = &bSmemTile[load_stage_idx][warp_col * kWN];

#pragma unroll
        for (int a = 0; a < kWM_ITER; a++) {
          FETCH_FLOAT4(aFragment[dot_product_idx % 2][a * kTM]) = FETCH_FLOAT4(
              aSmemWarpTile[thread_row * kTM + dot_product_idx * kBM +
                            a * warpMMATileM]);
        }

#pragma unroll
        for (int b = 0; b < kWN_ITER; b++) {
          FETCH_FLOAT4(bFragment[dot_product_idx % 2][b * kTN]) = FETCH_FLOAT4(
              bSmemWarpTile[thread_col * kTN + dot_product_idx * kBN +
                            b * warpMMATileN]);
        }
#pragma unroll
        for (uint32_t a = 0; a < kWM_ITER; a++) {
          for (uint32_t b = 0; b < kWN_ITER; b++) {
            for (uint32_t reg_idx_a = 0; reg_idx_a < kTM; reg_idx_a++) {
              for (uint32_t reg_idx_b = 0; reg_idx_b < kTN; reg_idx_b++) {
                result[(a * kTM + reg_idx_a) * kWN_ITER * kTN + b * kTN +
                       reg_idx_b] +=
                    aFragment[(dot_product_idx - 1) % 2][a * kTM + reg_idx_a] *
                    bFragment[(dot_product_idx - 1) % 2][b * kTN + reg_idx_b];
              }
            }
          }
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < load_a_iters; i++) {
        uint32_t load_smem_a = load_smem_a_col * kBM + load_smem_a_row +
                               i * load_smem_A_row_stride;
        aSmemTile[write_stage_idx][load_smem_a] = ldg_a[i];
      }
#pragma unroll
      for (uint32_t i = 0; i < load_b_iters; i++) {
        uint32_t load_smem_b =
            (load_smem_b_row + i * load_smem_B_row_stride) * kBN +
            load_smem_b_col;
        bSmemTile[write_stage_idx][load_smem_b] = ldg_b[i];
      }

      __syncthreads();

      float *aSmemWarpTile = &aSmemTile[write_stage_idx][warp_row * kWM];
      float *bSmemWarpTile = &bSmemTile[write_stage_idx][warp_col * kWN];

#pragma unroll
      for (int a = 0; a < kWM_ITER; a++) {
        FETCH_FLOAT4(aFragment[0][a * kTM]) = FETCH_FLOAT4(
            aSmemWarpTile[thread_row * kTM + 0 * kBM + a * warpMMATileM]);
      }
#pragma unroll
      for (int b = 0; b < kWN_ITER; b++) {
        FETCH_FLOAT4(bFragment[0][b * kTN]) = FETCH_FLOAT4(
            bSmemWarpTile[thread_col * kTN + 0 * kBN + b * warpMMATileN]);
      }

      write_stage_idx ^= 1;
#pragma unroll
      for (uint32_t a = 0; a < kWM_ITER; a++) {
        for (uint32_t b = 0; b < kWN_ITER; b++) {
          for (uint32_t reg_idx_a = 0; reg_idx_a < kTM; reg_idx_a++) {
            for (uint32_t reg_idx_b = 0; reg_idx_b < kTN; reg_idx_b++) {
              result[(a * kTM + reg_idx_a) * kWN_ITER * kTN + b * kTN +
                     reg_idx_b] +=
                  aFragment[(kBK - 1) % 2][a * kTM + reg_idx_a] *
                  bFragment[(kBK - 1) % 2][b * kTN + reg_idx_b];
            }
          }
        }
      }
    }
    uint32_t load_stage_idx = (k_iter - 1) % 2;
#pragma unroll
    for (uint32_t dot_product_idx = 1; dot_product_idx < kBK;
         ++dot_product_idx) {
      float *aSmemWarpTile = &aSmemTile[load_stage_idx][warp_row * kWM];
      float *bSmemWarpTile = &bSmemTile[load_stage_idx][warp_col * kWN];

#pragma unroll
      for (int a = 0; a < kWM_ITER; a++) {
        FETCH_FLOAT4(aFragment[dot_product_idx % 2][a * kTM]) = FETCH_FLOAT4(
            aSmemWarpTile[thread_row * kTM + dot_product_idx * kBM +
                          a * warpMMATileM]);
      }

#pragma unroll
      for (int b = 0; b < kWN_ITER; b++) {
        FETCH_FLOAT4(bFragment[dot_product_idx % 2][b * kTN]) = FETCH_FLOAT4(
            bSmemWarpTile[thread_col * kTN + dot_product_idx * kBN +
                          b * warpMMATileN]);
      }
#pragma unroll
      for (uint32_t a = 0; a < kWM_ITER; a++) {
        for (uint32_t b = 0; b < kWN_ITER; b++) {
          for (uint32_t reg_idx_a = 0; reg_idx_a < kTM; reg_idx_a++) {
            for (uint32_t reg_idx_b = 0; reg_idx_b < kTN; reg_idx_b++) {
              result[(a * kTM + reg_idx_a) * kWN_ITER * kTN + b * kTN +
                     reg_idx_b] +=
                  aFragment[(dot_product_idx - 1) % 2][a * kTM + reg_idx_a] *
                  bFragment[(dot_product_idx - 1) % 2][b * kTN + reg_idx_b];
            }
          }
        }
      }
    }
#pragma unroll
    for (uint32_t a = 0; a < kWM_ITER; a++) {
      for (uint32_t b = 0; b < kWN_ITER; b++) {
        for (uint32_t reg_idx_a = 0; reg_idx_a < kTM; reg_idx_a++) {
          for (uint32_t reg_idx_b = 0; reg_idx_b < kTN; reg_idx_b++) {
            result[(a * kTM + reg_idx_a) * kWN_ITER * kTN + b * kTN +
                   reg_idx_b] += aFragment[(kBK - 1) % 2][a * kTM + reg_idx_a] *
                                 bFragment[(kBK - 1) % 2][b * kTN + reg_idx_b];
          }
        }
      }
    }
  }
  // epilogue
  {
    float *final_C = &C[warp_row * kWM * N + warp_col * kWN];
#pragma unroll
    for (uint32_t a = 0; a < kWM_ITER; a++) {
      for (uint32_t b = 0; b < kWN_ITER; b++) {
        for (uint32_t reg_idx_a = 0; reg_idx_a < kTM; reg_idx_a++) {
          for (uint32_t reg_idx_b = 0; reg_idx_b < kTN; reg_idx_b++) {
            uint32_t row_offset =
                a * warpMMATileM + thread_row * kTM + reg_idx_a;
            uint32_t col_offset =
                b * warpMMATileN + thread_col * kTN + reg_idx_b;
            if (row_offset + warp_row * kWM + blockIdx.y * kBM < M &&
                col_offset + warp_col * kWN + blockIdx.x * kBN < N) {
              final_C[row_offset * N + col_offset] =
                  alpha * result[(a * kTM + reg_idx_a) * kWN_ITER * kTN +
                                 b * kTN + reg_idx_b] +
                  beta * final_C[row_offset * N + col_offset];
            }
          }
        }
      }
    }
  }
}