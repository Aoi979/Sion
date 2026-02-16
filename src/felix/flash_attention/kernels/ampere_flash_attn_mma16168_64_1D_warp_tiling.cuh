#pragma once
#include <cstdint>
#include <utils/utils.h>
#include <utils/swizzle.hpp>
#include <type_traits>
template <typename>
inline constexpr bool dependent_false_v = false;

template <int HEAD_DIM>
__device__ __forceinline__ uint32_t fa_swizzle(uint32_t i) {
    if constexpr (HEAD_DIM == 64) {
        return Swizzle<3, 4, 3>::apply(i);
    } else if constexpr (HEAD_DIM == 128) {
        return Swizzle<4, 4, 4>::apply(i);
    } else {
        static_assert(dependent_false_v<std::integral_constant<int, HEAD_DIM>>,
                      "HEAD_DIM is not supported");
        return i;
    }
}


template<int const HEAD_DIM, int const STAGE, int const Bc = 64>
__global__ void ampere_flash_attn_mma16168_64_1D_warp_tiling(half *Q, half *K, half *V, half *O,uint32_t heads,
                              uint32_t QKV_seqlen) {
    constexpr uint32_t MMA_M = 16;
    constexpr uint32_t MMA_N = 8;
    constexpr uint32_t MMA_K = 16;

    constexpr uint32_t WARP_NUM_SEQLEN_QS = 4;
    constexpr uint32_t WARP_NUM_SEQLEN_K = 1;
    // static assertions and constexpr calculations
    static_assert(MMA_M == 16 && MMA_N == 8 && MMA_K == 16);
    constexpr uint32_t Br = 64;
    // 1
    static_assert(Br % (MMA_M * WARP_NUM_SEQLEN_QS) == 0,
                  "Br must be divisible by MMA_M * WARP_NUM_SEQLEN_QS");

    constexpr uint32_t WARP_ITER_SEQLEN_QS = Br / (MMA_M * WARP_NUM_SEQLEN_QS);

    // 8
    static_assert(Bc % (MMA_N * WARP_NUM_SEQLEN_K) == 0,
                  "Bc must be divisible by MMA_N * WARP_NUM_SEQLEN_K");                  
    constexpr uint32_t WARP_ITER_SEQLEN_K = Bc / (MMA_N * WARP_NUM_SEQLEN_K);

    // 8
    static_assert(HEAD_DIM % (MMA_N * WARP_NUM_SEQLEN_K) == 0,
                  "HEAD_DIM must be divisible by MMA_N * WARP_NUM_SEQLEN_K");
    constexpr uint32_t WARP_ITER_HEAD_DIM_V =
            HEAD_DIM / (MMA_N * WARP_NUM_SEQLEN_K);
    static_assert(HEAD_DIM % MMA_K == 0, "HEAD_DIM must be divisible by MMA_K");
    constexpr uint32_t hidden_K_ITER = HEAD_DIM / MMA_K;

    static_assert(Bc % MMA_K == 0, "Bc must be divisible by MMA_K");
    constexpr uint32_t hidden_Bc_ITER = Bc / MMA_K;

    uint32_t QKV_HEADS = heads;
    uint32_t Tc = QKV_seqlen / Bc;

    constexpr uint32_t THREAD_SIZE =
            WARP_NUM_SEQLEN_K * WARP_NUM_SEQLEN_QS * WARP_SIZE;

    float scale = 1.0f / sqrt((float) HEAD_DIM);

    uint32_t QKV_batch_id = blockIdx.z;
    uint32_t QKV_head_id = blockIdx.y;
    uint32_t Q_tile_id = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    uint32_t warp_seqlen_qs_id = warp_id;
    uint32_t warp_seqlen_k_id = 0;

    uint32_t QKV_HEAD_SIZE = QKV_seqlen * HEAD_DIM;
    uint32_t BATCH_SIZE = QKV_HEADS * QKV_HEAD_SIZE;

    // Q, K, V, O [seqlen, head_dim]
    uint32_t Q_gmem_offset = QKV_batch_id * BATCH_SIZE +
                             QKV_head_id * QKV_HEAD_SIZE +
                             Q_tile_id * Br * HEAD_DIM;

    uint32_t K_gmem_offset =
            QKV_batch_id * BATCH_SIZE + QKV_head_id * QKV_HEAD_SIZE;

    uint32_t V_gmem_offset = K_gmem_offset;

    uint32_t O_gmem_offset = Q_gmem_offset;

    // LDG128 == 8 half
    constexpr uint32_t VECSIZE = 8;
    constexpr uint32_t smem_Q_row_thread_num = HEAD_DIM / VECSIZE;

    uint32_t load_smem_Q_Br = tid / smem_Q_row_thread_num;
    uint32_t load_smem_Q_d = (tid % smem_Q_row_thread_num) * VECSIZE;

    constexpr uint32_t load_smem_Q_stride = (THREAD_SIZE * VECSIZE) / HEAD_DIM;

    constexpr uint32_t smem_KV_row_thread_num = HEAD_DIM / VECSIZE;

    uint32_t load_smem_KV_Bc = tid / smem_KV_row_thread_num;
    uint32_t load_smem_KV_d = (tid % smem_KV_row_thread_num) * VECSIZE;

    constexpr uint32_t load_smem_KV_stride = (THREAD_SIZE * VECSIZE) / HEAD_DIM;

    extern __shared__ half sram[];

    constexpr uint32_t Q_tile_size = Br * HEAD_DIM;
    constexpr uint32_t KV_tile_size = Bc * HEAD_DIM;
    // constexpr uint32_t S_tile_size = Br * Bc;

    auto Q_tile_smem = sram;
    auto K_tile_smem = Q_tile_smem + Q_tile_size;
    auto V_tile_smem = K_tile_smem + STAGE * KV_tile_size;

    uint32_t Q_tile_smem_address = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t K_tile_smem_address = __cvta_generic_to_shared(K_tile_smem);
    uint32_t V_tile_smem_address = __cvta_generic_to_shared(V_tile_smem);

    float lane_Bc_max_old[WARP_ITER_SEQLEN_QS][2];
    fill_2D_regs<float, WARP_ITER_SEQLEN_QS, 2>(lane_Bc_max_old, -INFINITY);

    float lane_Bc_sum_old[WARP_ITER_SEQLEN_QS][2]={};


    using REG_SIZE_T = uint32_t;

    // 16 * 16 * 2 / (32 * 4) = 4
    REG_SIZE_T R_Q[WARP_ITER_SEQLEN_QS][4];
    // 16 * 8
    REG_SIZE_T R_K[WARP_ITER_SEQLEN_K][2];
    // 16 * 8
    REG_SIZE_T R_V[WARP_ITER_HEAD_DIM_V][2];
    // 16 * 8
    REG_SIZE_T R_S[WARP_ITER_SEQLEN_QS][WARP_ITER_SEQLEN_K][2] = {};
    // 16 * 8
    REG_SIZE_T R_O[WARP_ITER_SEQLEN_QS][WARP_ITER_HEAD_DIM_V][2] = {};
    // 16 * 8
    REG_SIZE_T R_Final[WARP_ITER_SEQLEN_QS][WARP_ITER_HEAD_DIM_V][2] = {};

    // load Q, gmem to smem, only once
    {
#pragma unroll
        for (int i = 0; i < Br / load_smem_Q_stride; i++) {
            uint32_t load_gmem_Q =
                    Q_gmem_offset + (load_smem_Q_Br + i * load_smem_Q_stride) * HEAD_DIM +
                    load_smem_Q_d;
            uint32_t load_smem_Q =
                    Q_tile_smem_address +
                    fa_swizzle<HEAD_DIM>(((load_smem_Q_Br + i * load_smem_Q_stride) * HEAD_DIM + load_smem_Q_d) * sizeof(half));
            CP_ASYNC_CG(load_smem_Q, &Q[load_gmem_Q], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    if constexpr (STAGE > 1) {
#pragma unroll
        for (int stage = 0; stage < STAGE - 1; stage++) {
#pragma unroll
            for (int i = 0; i < Bc / load_smem_KV_stride; i++) {
                uint32_t K_Bc_gmem_offset = stage * Bc;
                uint32_t load_gmem_K = K_gmem_offset + K_Bc_gmem_offset * HEAD_DIM +
                                       load_smem_KV_Bc * HEAD_DIM +
                                       i * load_smem_KV_stride * HEAD_DIM +
                                       load_smem_KV_d;
                uint32_t load_smem_K =
                        K_tile_smem_address +
                        fa_swizzle<HEAD_DIM>((stage * KV_tile_size + load_smem_KV_Bc * HEAD_DIM +
                         i * load_smem_KV_stride * HEAD_DIM + load_smem_KV_d) *
                        sizeof(half));
                CP_ASYNC_CG(load_smem_K, &K[load_gmem_K], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        // (STAGE - 1) - 1 = STAGE - 2
        // wait Q and 1 K stage
        CP_ASYNC_WAIT_GROUP(STAGE - 2);
        __syncthreads();
    }
// #pragma unroll
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; tile_K_seqlen++) {
        uint32_t K_smem_select = (tile_K_seqlen % STAGE);
        uint32_t K_smem_select_next = (tile_K_seqlen + (STAGE - 1)) % STAGE;

        if constexpr (STAGE > 1) {
            // load V asynchronously
            {
#pragma unroll
                for (int i = 0; i < Bc / load_smem_KV_stride; i++) {
                    uint32_t V_Bc_gmem_offset = tile_K_seqlen * Bc;
                    uint32_t load_gmem_V = V_gmem_offset + V_Bc_gmem_offset * HEAD_DIM +
                                           load_smem_KV_Bc * HEAD_DIM +
                                           i * load_smem_KV_stride * HEAD_DIM +
                                           load_smem_KV_d;
                    uint32_t load_smem_V =
                            V_tile_smem_address +
                            fa_swizzle<HEAD_DIM>((load_smem_KV_Bc * HEAD_DIM + i * load_smem_KV_stride * HEAD_DIM +
                             load_smem_KV_d) *
                            sizeof(half));
                    CP_ASYNC_CG(load_smem_V, &V[load_gmem_V], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }

            // NOTE: load next stage (only STGAE == 2) asynchronously
            if ((tile_K_seqlen + 1) < Tc) {
#pragma unroll
                for (int i = 0; i < Bc / load_smem_KV_stride; i++) {
                    uint32_t K_Bc_gmem_offset = (tile_K_seqlen + 1) * Bc;
                    uint32_t load_gmem_K = K_gmem_offset + K_Bc_gmem_offset * HEAD_DIM +
                                           load_smem_KV_Bc * HEAD_DIM +
                                           i * load_smem_KV_stride * HEAD_DIM +
                                           load_smem_KV_d;
                    uint32_t load_smem_K =
                            K_tile_smem_address +
                            fa_swizzle<HEAD_DIM>((K_smem_select_next * KV_tile_size + load_smem_KV_Bc * HEAD_DIM +
                             i * load_smem_KV_stride * HEAD_DIM + load_smem_KV_d) *
                            sizeof(half));
                    CP_ASYNC_CG(load_smem_K, &K[load_gmem_K], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
        } else {
            // stage == 1
            // ..........
            // ..........
        }

        fill_3D_regs<uint32_t, WARP_ITER_SEQLEN_QS, WARP_ITER_SEQLEN_K, 2>(R_S, 0);
#pragma unroll
        for (int bk_d = 0; bk_d < hidden_K_ITER; bk_d++) {
            // Q | smem2register
#pragma unroll
            for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
                uint32_t Q_smem_warp_offset =
                        (warp_seqlen_qs_id * MMA_M * WARP_ITER_SEQLEN_QS) + i * MMA_M;
                uint32_t lane_Q_smem_Br = Q_smem_warp_offset + lane_id % MMA_M;
                uint32_t lane_Q_smem_d = bk_d * MMA_K + (lane_id / MMA_M) * 8;
                uint32_t lane_Q_smem_address =
                        Q_tile_smem_address +
                        fa_swizzle<HEAD_DIM>(((lane_Q_smem_Br * HEAD_DIM) + lane_Q_smem_d) * sizeof(half));
                LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                            lane_Q_smem_address);
            }
            // K | smem2register
#pragma unroll
            for (int i = 0; i < WARP_ITER_SEQLEN_K; i++) {
                uint32_t K_smem_warp_offset =
                        warp_seqlen_k_id * MMA_N * WARP_ITER_SEQLEN_K + i * MMA_N;
                uint32_t lane_K_smem_Bc = K_smem_warp_offset + lane_id % MMA_N;
                uint32_t lane_K_smem_d = bk_d * MMA_K + ((lane_id / MMA_N) % 2) * 8;
                uint32_t lane_K_smem_address =
                        K_tile_smem_address + fa_swizzle<HEAD_DIM>((K_smem_select * KV_tile_size +
                                               lane_K_smem_Bc * HEAD_DIM + lane_K_smem_d) *
                        sizeof(half));
                LDMATRIX_X2(R_K[i][0], R_K[i][1], lane_K_smem_address);
            }
#pragma unroll
            for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
#pragma unroll
                for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
                    HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                              R_Q[i][3], R_K[j][0], R_K[j][1], R_S[i][j][0],
                              R_S[i][j][1]);
                }
            }
        }

        float lane_row_max_new[WARP_ITER_SEQLEN_QS][2];
        fill_2D_regs<float, WARP_ITER_SEQLEN_QS, 2>(lane_row_max_new, -INFINITY);
        float lane_row_sum_new[WARP_ITER_SEQLEN_QS][2] = {};
        // warp level reduce max and store to smem
#pragma unroll
        for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
                float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));

                float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
                float tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
                float tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
                lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
                lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
            }
            lane_row_max_new[i][0] =
                    warp_reduce_max<float, 4>(lane_row_max_new[i][0]);
            lane_row_max_new[i][1] =
                    warp_reduce_max<float, 4>(lane_row_max_new[i][1]);
        }

#pragma unroll
        for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
            float Bc_row_max_0 = lane_row_max_new[i][0];
            float Bc_row_max_1 = lane_row_max_new[i][1];

            float Bc_row_max_old_0 = lane_Bc_max_old[i][0];
            float Bc_row_max_old_1 = lane_Bc_max_old[i][1];

            Bc_row_max_0 = max(Bc_row_max_old_0, Bc_row_max_0);
            Bc_row_max_1 = max(Bc_row_max_old_1, Bc_row_max_1);
#pragma unroll
            for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
                float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
                float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
                t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, -Bc_row_max_0));
                t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, -Bc_row_max_0));
                t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, -Bc_row_max_1));
                t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, -Bc_row_max_1));

                lane_row_sum_new[i][0] += t_reg_S_0.x + t_reg_S_0.y;
                lane_row_sum_new[i][1] += t_reg_S_1.x + t_reg_S_1.y;

                HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
                HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
            }

            // warp level reduce sum and store to smem
            lane_row_sum_new[i][0] =
                    warp_reduce_sum<float, 4>(lane_row_sum_new[i][0]);
            lane_row_sum_new[i][1] =
                    warp_reduce_sum<float, 4>(lane_row_sum_new[i][1]);
        }

        if constexpr (STAGE > 1) {
            if (tile_K_seqlen + 1 < Tc) {
                CP_ASYNC_WAIT_GROUP(1);
            } else {
                CP_ASYNC_WAIT_GROUP(0);
            }
        } else {
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();
        fill_3D_regs<uint32_t, WARP_ITER_SEQLEN_QS, WARP_ITER_HEAD_DIM_V, 2>(R_O,0);
#pragma unroll
        for (int bk_Bc = 0; bk_Bc < hidden_Bc_ITER; bk_Bc++) {
#pragma unroll
            for (int i = 0; i < WARP_ITER_HEAD_DIM_V; i++) {
                uint32_t V_smem_warp_offset =
                        warp_seqlen_k_id * MMA_N * WARP_ITER_HEAD_DIM_V + i * MMA_N;
                uint32_t lane_V_smem_Bc = bk_Bc * MMA_K + lane_id % MMA_K;
                uint32_t lane_V_smem_d = V_smem_warp_offset;
                uint32_t lane_V_smem_address =
                        V_tile_smem_address +
                        fa_swizzle<HEAD_DIM>((lane_V_smem_Bc * HEAD_DIM + lane_V_smem_d) * sizeof(half));
                LDMATRIX_X2_T(R_V[i][0], R_V[i][1], lane_V_smem_address);
            }
            int idx = bk_Bc * 2;
#pragma unroll
            for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
#pragma unroll
                for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
                    HMMA16816(R_O[i][j][0], R_O[i][j][1], R_S[i][idx][0], R_S[i][idx][1], R_S[i][idx + 1][0],
                              R_S[i][idx + 1][1], R_V[j][0], R_V[j][1], R_O[i][j][0],
                              R_O[i][j][1]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
            float Bc_row_max_0 = lane_row_max_new[i][0];
            float Bc_row_max_1 = lane_row_max_new[i][1];
            float Bc_row_sum_0 = lane_row_sum_new[i][0];
            float Bc_row_sum_1 = lane_row_sum_new[i][1];
            
            float Bc_row_max_old_0 = lane_Bc_max_old[i][0];
            float Bc_row_max_old_1 = lane_Bc_max_old[i][1];

            Bc_row_max_0 = max(Bc_row_max_old_0, Bc_row_max_0);
            Bc_row_max_1 = max(Bc_row_max_old_1, Bc_row_max_1);


            Bc_row_max_old_0 = (tile_K_seqlen > 0 ? Bc_row_max_old_0 : Bc_row_max_0);
            Bc_row_max_old_1 = (tile_K_seqlen > 0 ? Bc_row_max_old_1 : Bc_row_max_1);


            float rescale_o_factor_0 = __expf(Bc_row_max_old_0 - Bc_row_max_0);

            float rescale_o_factor_1 = __expf(Bc_row_max_old_1 - Bc_row_max_1);

#pragma unroll
            for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
                float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0]));
                float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1]));
                float2 t_reg_Final_0 = __half22float2(HALF2(R_Final[i][j][0]));
                float2 t_reg_Final_1 = __half22float2(HALF2(R_Final[i][j][1]));

                t_reg_Final_0.x =
                        __fmaf_rn(rescale_o_factor_0, t_reg_Final_0.x, t_reg_O_0.x);
                t_reg_Final_0.y =
                        __fmaf_rn(rescale_o_factor_0, t_reg_Final_0.y, t_reg_O_0.y);
                t_reg_Final_1.x =
                        __fmaf_rn(rescale_o_factor_1, t_reg_Final_1.x, t_reg_O_1.x);
                t_reg_Final_1.y =
                        __fmaf_rn(rescale_o_factor_1, t_reg_Final_1.y, t_reg_O_1.y);


                HALF2(R_Final[i][j][0]) = __float22half2_rn(t_reg_Final_0);
                HALF2(R_Final[i][j][1]) = __float22half2_rn(t_reg_Final_1);
            }
            float Bc_row_sum_old_0 = lane_Bc_sum_old[i][0];
            float Bc_row_sum_old_1 = lane_Bc_sum_old[i][1];

            lane_Bc_sum_old[i][0] =
                    __fmaf_rn(rescale_o_factor_0, Bc_row_sum_old_0, Bc_row_sum_0);
            lane_Bc_sum_old[i][1] =
                    __fmaf_rn(rescale_o_factor_1, Bc_row_sum_old_1, Bc_row_sum_1);
            lane_Bc_max_old[i][0] = Bc_row_max_0;
            lane_Bc_max_old[i][1] = Bc_row_max_1;
        }
        if constexpr (STAGE > 1) {
            if ((tile_K_seqlen + 1) < Tc) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }

        }
    }


#pragma unroll
    for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
        float rescale_factor_0 = __frcp_rn(lane_Bc_sum_old[i][0]);
        float rescale_factor_1 = __frcp_rn(lane_Bc_sum_old[i][1]);

#pragma unroll
        for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
            float2 t_reg_Final_0 = __half22float2(HALF2(R_Final[i][j][0]));
            float2 t_reg_Final_1 = __half22float2(HALF2(R_Final[i][j][1]));
            t_reg_Final_0.x = rescale_factor_0 * t_reg_Final_0.x;
            t_reg_Final_0.y = rescale_factor_0 * t_reg_Final_0.y;
            t_reg_Final_1.x = rescale_factor_1 * t_reg_Final_1.x;
            t_reg_Final_1.y = rescale_factor_1 * t_reg_Final_1.y;
            HALF2(R_Final[i][j][0]) = __float22half2_rn(t_reg_Final_0);
            HALF2(R_Final[i][j][1]) = __float22half2_rn(t_reg_Final_1);
        }
    }
#pragma unroll
    for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
#pragma unroll
        for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
            REG_SIZE_T R_T[2][4];
            R_T[0][0] = R_Final[i][j][0];
            R_T[1][0] = R_Final[i][j][1];
            R_T[0][1] = __shfl_sync((0xffffffff), R_Final[i][j][0], lane_id + 1, 4);
            R_T[0][2] = __shfl_sync((0xffffffff), R_Final[i][j][0], lane_id + 2, 4);
            R_T[0][3] = __shfl_sync((0xffffffff), R_Final[i][j][0], lane_id + 3, 4);
            R_T[1][1] = __shfl_sync((0xffffffff), R_Final[i][j][1], lane_id + 1, 4);
            R_T[1][2] = __shfl_sync((0xffffffff), R_Final[i][j][1], lane_id + 2, 4);
            R_T[1][3] = __shfl_sync((0xffffffff), R_Final[i][j][1], lane_id + 3, 4);

            if (lane_id % 4 == 0) {
                uint32_t store_O_gmem_Br =
                        warp_seqlen_qs_id * (MMA_M * WARP_ITER_SEQLEN_QS) + i * MMA_M +
                        lane_id / 4;
                uint32_t store_O_gmem_d =
                        warp_seqlen_k_id * (MMA_N * WARP_ITER_HEAD_DIM_V) + j * MMA_N;
                uint32_t store_gmem_O_address_0 =
                        O_gmem_offset + (store_O_gmem_Br + 0) * HEAD_DIM + store_O_gmem_d;
                uint32_t store_gmem_O_address_1 =
                        O_gmem_offset + (store_O_gmem_Br + 8) * HEAD_DIM + store_O_gmem_d;
                LDST128BITS(O[store_gmem_O_address_0]) = LDST128BITS(R_T[0][0]);
                LDST128BITS(O[store_gmem_O_address_1]) = LDST128BITS(R_T[1][0]);
            }
        }
    }
} 