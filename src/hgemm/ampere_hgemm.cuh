#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

// ampere_hgemm_128x128
// cudaFuncSetAttribute(this, cudaFuncAttributeMaxDynamicSharedMemorySize, 37888);
// 4 warps per block
// balanced workload
template<int const BM = 128, int const BN = 128, int const bK = 32, int const WM = 64, int const WN = 64>
__global__ void ampere_hgemm_128x128(half *A, half *B, half *C, int M, int N, int K) {
    uint32_t warp_id = threadIdx.x / 32;

    constexpr uint32_t A_PADDING = 8;
    constexpr uint32_t B_PADDING = 8;
    extern __shared__ half smem[];
    using array2d_a = half(*)[BM * (bK + A_PADDING)];
    using array2d_b = half(*)[(BN + B_PADDING) * bK];
    auto a_smem = reinterpret_cast<array2d_a>(smem);
    auto b_smem = reinterpret_cast<array2d_b>(smem + 2 * (BM * (bK + A_PADDING)));
    constexpr uint32_t bK_T = bK + A_PADDING;
    constexpr uint32_t BN_T = BN + B_PADDING;
    constexpr uint32_t FRAGMENT_SIZE = 16;
    constexpr uint32_t WARP_SIZE = 32;

    constexpr uint32_t bk_iter = bK / FRAGMENT_SIZE;
    constexpr uint32_t wm_iter = WM / FRAGMENT_SIZE;
    constexpr uint32_t wn_iter = WN / FRAGMENT_SIZE;

    uint32_t warp_row = warp_id / (BN / WN);
    uint32_t warp_col = warp_id % (BN / WN);

    // LDG 128, 8 * 16bit
    constexpr uint32_t VEC_SIZE = 8;
    constexpr uint32_t threads_per_block = ((BM * BN) / (WM * WN)) * WARP_SIZE;
    constexpr uint32_t load_a_per_thread = ((BM * bK) / threads_per_block) / VEC_SIZE;
    constexpr uint32_t load_b_per_thread = ((BN * bK) / threads_per_block) / VEC_SIZE;
    // 32 * 64
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][wm_iter];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][wn_iter];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[wm_iter][wn_iter];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
#pragma unroll
    for (uint32_t m = 0; m < wm_iter; m++) {
#pragma unroll
        for (uint32_t n = 0; n < wn_iter; n++) {
            wmma::fill_fragment(frag_c[m][n], 0.0f);
        }
    }
    int a_smem_addr = __cvta_generic_to_shared(&a_smem[0][0]);
    int b_smem_addr = __cvta_generic_to_shared(&b_smem[0][0]);

    uint32_t k_iter = K / bK;


#pragma unroll
    for (uint32_t a = 0; a < load_a_per_thread; a++) {
        uint32_t col_offset = threadIdx.x % (bK / VEC_SIZE);
        uint32_t row_offset = threadIdx.x / (bK / VEC_SIZE);
        asm (
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"((int) (a_smem_addr + (int) (
                             row_offset * bK_T + col_offset * VEC_SIZE + a * ((threads_per_block * VEC_SIZE) / bK) *
                             bK_T) *
                         sizeof(half))), "l"(&A[row_offset * K + col_offset * VEC_SIZE + a * (
                                                    (threads_per_block * VEC_SIZE) / bK) * K])
        );
    }
#pragma unroll
    for (uint32_t b = 0; b < load_b_per_thread; b++) {
        uint32_t col_offset = threadIdx.x % (BN / VEC_SIZE);
        uint32_t row_offset = threadIdx.x / (BN / VEC_SIZE);
        asm (
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"((int) (b_smem_addr + (int) (
                             row_offset * BN_T + col_offset * VEC_SIZE + b * ((threads_per_block * VEC_SIZE) / BN) *
                             BN_T) *
                         sizeof(half))), "l"(&B[row_offset * N + col_offset * VEC_SIZE + b * (
                                                    (threads_per_block * VEC_SIZE) / BN) * N])
        );
    }
    A += bK;
    B += bK * N;
    asm(
        "cp.async.commit_group;\n"
        :
        :
    );
    asm(
        "cp.async.wait_group 0;\n"
        :
        :
    );
    __syncthreads();
#pragma unroll
    for (uint32_t m = 0; m < wm_iter; m++) {
        wmma::load_matrix_sync(frag_a[0][m],
                               &a_smem[0][
                                   warp_row * WM * bK_T + m * FRAGMENT_SIZE * bK_T + 0 * FRAGMENT_SIZE],
                               bK_T);
    }
#pragma unroll
    for (uint32_t n = 0; n < wn_iter; n++) {
        wmma::load_matrix_sync(frag_b[0][n],
                               &b_smem[0][
                                   warp_col * WN + 0 * FRAGMENT_SIZE * BN_T + n * FRAGMENT_SIZE], BN_T);
    }
    uint32_t write_stage_idx = 1;


#pragma unroll
    for (uint32_t k = 1; k < k_iter; k++) {
        int a_smem_addr = __cvta_generic_to_shared(&a_smem[write_stage_idx][0]);
        int b_smem_addr = __cvta_generic_to_shared(&b_smem[write_stage_idx][0]);

#pragma unroll
        for (uint32_t a = 0; a < load_a_per_thread; a++) {
            uint32_t col_offset = threadIdx.x % (bK / VEC_SIZE);
            uint32_t row_offset = threadIdx.x / (bK / VEC_SIZE);
            asm (
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"((int) (a_smem_addr + (int) (
                                 row_offset * bK_T + col_offset * VEC_SIZE + a * ((threads_per_block * VEC_SIZE) / bK) *
                                 bK_T) *
                             sizeof(half))), "l"(&A[row_offset * K + col_offset * VEC_SIZE + a * (
                                                        (threads_per_block * VEC_SIZE) / bK) * K])
            );
        }
#pragma unroll
        for (uint32_t b = 0; b < load_b_per_thread; b++) {
            uint32_t col_offset = threadIdx.x % (BN / VEC_SIZE);
            uint32_t row_offset = threadIdx.x / (BN / VEC_SIZE);
            asm (
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"((int) (b_smem_addr + (int) (
                                 row_offset * BN_T + col_offset * VEC_SIZE + b * ((threads_per_block * VEC_SIZE) / BN) *
                                 BN_T) *
                             sizeof(half))), "l"(&B[row_offset * N + col_offset * VEC_SIZE + b * (
                                                        (threads_per_block * VEC_SIZE) / BN) * N])
            );
        }
        A += bK;
        B += bK * N;
        uint32_t load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (uint32_t bk = 1; bk < bk_iter; bk++) {
#pragma unroll
            for (uint32_t m = 0; m < wm_iter; m++) {
                wmma::load_matrix_sync(frag_a[bk % 2][m],
                                       &a_smem[load_stage_idx][
                                           warp_row * WM * bK_T + m * FRAGMENT_SIZE * bK_T + bk * FRAGMENT_SIZE],
                                       bK_T);
            }
#pragma unroll
            for (uint32_t n = 0; n < wn_iter; n++) {
                wmma::load_matrix_sync(frag_b[bk % 2][n],
                                       &b_smem[load_stage_idx][
                                           warp_col * WN + bk * FRAGMENT_SIZE * BN_T + n * FRAGMENT_SIZE], BN_T);
            }

#pragma unroll
            for (uint32_t m = 0; m < wm_iter; m++) {
#pragma unroll
                for (uint32_t n = 0; n < wn_iter; n++) {
                    wmma::mma_sync(frag_c[m][n], frag_a[(bk - 1) % 2][m], frag_b[(bk - 1) % 2][n], frag_c[m][n]);
                }
            }
        }
        asm(
            "cp.async.commit_group;\n"
            :
            :
        );
        asm(
            "cp.async.wait_group 0;\n"
            :
            :
        );
#pragma unroll
        for (uint32_t m = 0; m < wm_iter; m++) {
#pragma unroll
            for (uint32_t n = 0; n < wn_iter; n++) {
                wmma::mma_sync(frag_c[m][n], frag_a[1][m], frag_b[1][n], frag_c[m][n]);
            }
        }
        __syncthreads();

#pragma unroll
        for (uint32_t m = 0; m < wm_iter; m++) {
            wmma::load_matrix_sync(frag_a[0][m],
                                   &a_smem[write_stage_idx][
                                       warp_row * WM * bK_T + m * FRAGMENT_SIZE * bK_T + 0 * FRAGMENT_SIZE],
                                   bK_T);
        }
#pragma unroll
        for (uint32_t n = 0; n < wn_iter; n++) {
            wmma::load_matrix_sync(frag_b[0][n],
                                   &b_smem[write_stage_idx][
                                       warp_col * WN + 0 * FRAGMENT_SIZE * BN_T + n * FRAGMENT_SIZE], BN_T);
        }

        write_stage_idx ^= 1;
    }

#pragma unroll
    for (uint32_t bk = 1; bk < bk_iter; bk++) {
#pragma unroll
        for (uint32_t m = 0; m < wm_iter; m++) {
            wmma::load_matrix_sync(frag_a[bk % 2][m],
                                   &a_smem[1][
                                       warp_row * WM * bK_T + m * FRAGMENT_SIZE * bK_T + bk * FRAGMENT_SIZE],
                                   bK_T);
        }
#pragma unroll
        for (uint32_t n = 0; n < wn_iter; n++) {
            wmma::load_matrix_sync(frag_b[bk % 2][n],
                                   &b_smem[1][
                                       warp_col * WN + bk * FRAGMENT_SIZE * BN_T + n * FRAGMENT_SIZE], BN_T);
        }

#pragma unroll
        for (uint32_t m = 0; m < wm_iter; m++) {
#pragma unroll
            for (uint32_t n = 0; n < wn_iter; n++) {
                wmma::mma_sync(frag_c[m][n], frag_a[(bk - 1) % 2][m], frag_b[(bk - 1) % 2][n], frag_c[m][n]);
            }
        }
    }

#pragma unroll
    for (uint32_t m = 0; m < wm_iter; m++) {
#pragma unroll
        for (uint32_t n = 0; n < wn_iter; n++) {
            wmma::mma_sync(frag_c[m][n], frag_a[1][m], frag_b[1][n], frag_c[m][n]);
        }
    }

#pragma unroll
    for (uint32_t m = 0; m < wm_iter; m++) {
#pragma unroll
        for (uint32_t n = 0; n < wn_iter; n++) {
            wmma::store_matrix_sync(&C[warp_row * WM * N + warp_col * WN + m * FRAGMENT_SIZE * N + n * FRAGMENT_SIZE],
                                    frag_c[m][n], N, wmma::mem_row_major);
        }
    }
}