#include <cstdint>
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])

// This is a specialized version that requires shape alignment and outperforms cublas.
template<int const BM = 128, int const BN = 128, int const TM = 4, int const TN = 4, int const WM = 32, int const WN =
        64, int const bK = 16, int const WM_ITER = 2, int const WN_ITER = 2>
__global__ void ampere_sgemm_128x128x16_aligned(int M, int N, int K, float alpha,
                                                     float const *A, float const *B,
                                                     float beta, float *C) {
    __shared__ float a_smem[2][bK * BM];
    __shared__ float b_smem[2][bK * BN];
    constexpr uint32_t VEC_SIZE = 4;
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t threads_per_block = ((BM * BN) / (WM * WN)) * WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float reg_a[2][2 * TM] = {0.0};
    float reg_b[2][2 * TN] = {0.0};
    float result[4 * TM * TN] = {0.0};
    constexpr uint32_t load_a_per_thread = ((bK * BM) / threads_per_block) / VEC_SIZE;
    constexpr uint32_t load_b_per_thread = ((bK * BN) / threads_per_block) / VEC_SIZE;
    constexpr uint32_t warps_per_row = BN / WN;
    uint32_t warp_row = warp_id / warps_per_row;
    uint32_t warp_col = warp_id % warps_per_row;
    constexpr uint32_t inner_N = WN / WN_ITER;
    constexpr uint32_t inner_M = WM / WM_ITER;
    uint32_t inner_row = lane_id / (inner_N / TN);
    uint32_t inner_col = lane_id % (inner_N / TN);
    uint32_t k_iter = K / bK;
    float *inner_A = &a_smem[0][warp_row * WM];
    float *inner_B = &b_smem[0][warp_col * WN];
    float4 ldg_a[load_a_per_thread];
    float4 ldg_b[load_b_per_thread];

#pragma unroll
    for (int a = 0; a < load_a_per_thread; ++a) {
        uint32_t smem_idx = ((threadIdx.x % (bK / VEC_SIZE)) * VEC_SIZE * BM) + (threadIdx.x / (bK / VEC_SIZE)) + (
                                (a * threads_per_block * VEC_SIZE) / bK);
        uint32_t global_a_idx = (smem_idx % BM) * K + (smem_idx / BM);
        ldg_a[0] = FETCH_CONST_FLOAT4(A[global_a_idx]);
        a_smem[0][smem_idx] = ldg_a[0].x;
        a_smem[0][smem_idx + BM] = ldg_a[0].y;
        a_smem[0][smem_idx + 2 * BM] = ldg_a[0].z;
        a_smem[0][smem_idx + 3 * BM] = ldg_a[0].w;
    }
#pragma unroll
    for (int b = 0; b < load_b_per_thread; ++b) {
        uint32_t smem_idx = threadIdx.x * VEC_SIZE + b * threads_per_block * VEC_SIZE;
        uint32_t global_b_idx = (smem_idx / BN) * N + (smem_idx % BN);
        FETCH_FLOAT4(b_smem[0][smem_idx]) = FETCH_CONST_FLOAT4(B[global_b_idx]);
    }

    __syncthreads();
#pragma unroll
    for (int a = 0; a < WM_ITER; a++) {
        for (int i = 0; i < TM / VEC_SIZE; i++) {
            FETCH_FLOAT4(reg_a[0][a * TM + i * VEC_SIZE]) = FETCH_FLOAT4(
                inner_A[inner_row * TM + i * VEC_SIZE + 0 * BM + a * inner_M]);
        }
    }
#pragma unroll
    for (int b = 0; b < WN_ITER; b++) {
        for (int i = 0; i < TN / VEC_SIZE; i++) {
            FETCH_FLOAT4(reg_b[0][b * TN + i * VEC_SIZE]) = FETCH_FLOAT4(
                inner_B[inner_col * TN + i * VEC_SIZE + 0 * BN + b * inner_N]);
        }
    }

    uint32_t write_stage_idx = 1;
    for (int k = 1; k < k_iter; ++k) {
#pragma unroll
        for (int a = 0; a < load_a_per_thread; ++a) {
            uint32_t smem_idx = ((threadIdx.x % (bK / VEC_SIZE)) * VEC_SIZE * BM) + (threadIdx.x / (bK / VEC_SIZE)) + (
                                    (a * threads_per_block * VEC_SIZE) / bK);
            uint32_t global_a_idx = (smem_idx % BM) * K + (smem_idx / BM) + k * bK;
            ldg_a[a] = FETCH_CONST_FLOAT4(A[global_a_idx]);
        }
#pragma unroll
        for (int b = 0; b < load_b_per_thread; ++b) {
            uint32_t smem_idx = threadIdx.x * VEC_SIZE + b * threads_per_block * VEC_SIZE;
            uint32_t global_b_idx = (smem_idx / BN) * N + (smem_idx % BN) + k * bK * N;
            ldg_b[b] = FETCH_CONST_FLOAT4(B[global_b_idx]);
        }


        uint32_t load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (uint32_t dot_product_idx = 1; dot_product_idx < bK; ++dot_product_idx) {
            float *inner_A = &a_smem[load_stage_idx][warp_row * WM];
            float *inner_B = &b_smem[load_stage_idx][warp_col * WN];
#pragma unroll
            for (int a = 0; a < WM_ITER; a++) {
                for (int i = 0; i < TM / VEC_SIZE; i++) {
                    FETCH_FLOAT4(reg_a[dot_product_idx % 2][a * TM + i * VEC_SIZE]) = FETCH_FLOAT4(
                        inner_A[inner_row * TM + i * VEC_SIZE + dot_product_idx * BM + a * inner_M]);
                }
            }
#pragma unroll
            for (int b = 0; b < WN_ITER; b++) {
                for (int i = 0; i < TN / VEC_SIZE; i++) {
                    FETCH_FLOAT4(reg_b[dot_product_idx % 2][b * TN + i * VEC_SIZE]) = FETCH_FLOAT4(
                        inner_B[inner_col * TN + i * VEC_SIZE + dot_product_idx * BN + b * inner_N]);
                }
            }
#pragma unroll
            for (int a = 0; a < WM_ITER; a++) {
                for (int b = 0; b < WN_ITER; b++) {
                    for (int reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                        for (int reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                            result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b] += reg_a[
                                (dot_product_idx - 1) % 2][
                                a * TM + reg_idx_a] * reg_b[(dot_product_idx - 1) % 2][b * TN + reg_idx_b];
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int a = 0; a < load_a_per_thread; ++a) {
            uint32_t smem_idx = ((threadIdx.x % (bK / VEC_SIZE)) * VEC_SIZE * BM) + (threadIdx.x / (bK / VEC_SIZE)) + (
                                    (a * threads_per_block * VEC_SIZE) / bK);
            float4 ldg_a_ = ldg_a[a];
            a_smem[write_stage_idx][smem_idx] = ldg_a_.x;
            a_smem[write_stage_idx][smem_idx + BM] = ldg_a_.y;
            a_smem[write_stage_idx][smem_idx + 2 * BM] = ldg_a_.z;
            a_smem[write_stage_idx][smem_idx + 3 * BM] = ldg_a_.w;
        }
#pragma unroll
        for (int b = 0; b < load_b_per_thread; ++b) {
            uint32_t smem_idx = threadIdx.x * VEC_SIZE + b * threads_per_block * VEC_SIZE;
            float4 ldg_b_ = ldg_b[b];
            FETCH_FLOAT4(b_smem[write_stage_idx][smem_idx]) = ldg_b_;
        }


        __syncthreads();

        float *inner_A = &a_smem[write_stage_idx][warp_row * WM];
        float *inner_B = &b_smem[write_stage_idx][warp_col * WN];
#pragma unroll
        for (int a = 0; a < WM_ITER; a++) {
            for (int i = 0; i < TM / VEC_SIZE; i++) {
                FETCH_FLOAT4(reg_a[0][a * TM + i * VEC_SIZE]) = FETCH_FLOAT4(
                    inner_A[inner_row * TM + i * VEC_SIZE + 0 * BM + a * inner_M]);
            }
        }
#pragma unroll
        for (int b = 0; b < WN_ITER; b++) {
            for (int i = 0; i < TN / VEC_SIZE; i++) {
                FETCH_FLOAT4(reg_b[0][b * TN + i * VEC_SIZE]) = FETCH_FLOAT4(
                    inner_B[inner_col * TN + i * VEC_SIZE + 0 * BN + b * inner_N]);
            }
        }

        write_stage_idx ^= 1;
#pragma unroll
        for (int a = 0; a < WM_ITER; a++) {
            for (int b = 0; b < WN_ITER; b++) {
                for (int reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                    for (int reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                        result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b] += reg_a[1][
                            a * TM + reg_idx_a] * reg_b[1][b * TN + reg_idx_b];
                    }
                }
            }
        }
    }
#pragma unroll
    for (uint32_t dot_product_idx = 1; dot_product_idx < bK; ++dot_product_idx) {
        float *inner_A = &a_smem[1][warp_row * WM];
        float *inner_B = &b_smem[1][warp_col * WN];
#pragma unroll
        for (int a = 0; a < WM_ITER; a++) {
            for (int i = 0; i < TM / VEC_SIZE; i++) {
                FETCH_FLOAT4(reg_a[dot_product_idx % 2][a * TM + i * VEC_SIZE]) = FETCH_FLOAT4(
                    inner_A[inner_row * TM + i * VEC_SIZE + dot_product_idx * BM + a * inner_M]);
            }
        }
#pragma unroll
        for (int b = 0; b < WN_ITER; b++) {
            for (int i = 0; i < TN / VEC_SIZE; i++) {
                FETCH_FLOAT4(reg_b[dot_product_idx % 2][b * TN + i * VEC_SIZE]) = FETCH_FLOAT4(
                    inner_B[inner_col * TN + i * VEC_SIZE + dot_product_idx * BN + b * inner_N]);
            }
        }
#pragma unroll
        for (int a = 0; a < WM_ITER; a++) {
            for (int b = 0; b < WN_ITER; b++) {
                for (int reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                    for (int reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                        result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b] += reg_a[
                            (dot_product_idx - 1) % 2][
                            a * TM + reg_idx_a] * reg_b[(dot_product_idx - 1) % 2][b * TN + reg_idx_b];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < WM_ITER; a++) {
        for (int b = 0; b < WN_ITER; b++) {
            for (int reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                for (int reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                    result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b] += reg_a[1][
                        a * TM + reg_idx_a] * reg_b[1][b * TN + reg_idx_b];
                }
            }
        }
    }
    float *final_C = &C[warp_row * WM * N + warp_col * WN];
#pragma unroll
    for (int a = 0; a < WM_ITER; a++) {
        for (int b = 0; b < WN_ITER; b++) {
            for (uint32_t reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                for (uint32_t reg_idx_b = 0; reg_idx_b < TN / VEC_SIZE; reg_idx_b++) {
                    uint32_t row_offset = a * inner_M + inner_row * TM + reg_idx_a;
                    uint32_t col_offset = b * inner_N + inner_col * TN + reg_idx_b * VEC_SIZE;
                    FETCH_FLOAT4(final_C[row_offset * N + col_offset]) =
                            FETCH_FLOAT4(
                                result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b * VEC_SIZE
                                ]);
                }
            }
        }
    }
}
