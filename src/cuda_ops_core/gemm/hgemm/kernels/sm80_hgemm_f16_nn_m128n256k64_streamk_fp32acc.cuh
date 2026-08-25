#pragma once

#include "../detail/sm80/splitk.cuh"

namespace cuda_ops_core::detail::sm80::streamk {

using namespace ::cuda_ops_core::detail::sm80::common;
using namespace ::cuda_ops_core::detail::sm80::tile;
namespace base = ::cuda_ops_core::detail::sm80::splitk::support;
namespace tile = ::cuda_ops_core::detail::sm80::tile_n256;

#include <stddef.h>
#include <stdint.h>

constexpr int kInvalidWork = -1;

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

struct StreamKSchedulePlan {
  int tiled_shape_m = 0;
  int tiled_shape_n = 0;
  int iters_per_tile = 0;
  int output_tile_count = 0;

  int dp_blocks = 0;
  int sk_tiles = 0;
  int sk_blocks = 0;
  int sk_iters_per_block = 0;
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

struct StreamKParams {
  int tiled_shape_n = 0;
  int iters_per_tile = 0;
  int dp_blocks = -1;
  int sk_tiles = 0;
  int sk_blocks = 0;
  int sk_iters_per_block = 0;
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

__host__ __device__ inline int get_tile_peer_count(
    int tile_id, int sk_tiles, int iters_per_tile, int sk_blocks,
    int sk_iters_per_block) {
  if (tile_id < 0 || tile_id >= sk_tiles || iters_per_tile <= 0 ||
      sk_blocks <= 0 || sk_iters_per_block <= 0) {
    return 0;
  }
  if (sk_tiles > INT32_MAX / iters_per_tile) {
    return 0;
  }
  int tile_begin = tile_id * iters_per_tile;
  int tile_end = tile_begin + iters_per_tile;
  int first = static_cast<int>(tile_begin / sk_iters_per_block);
  int end = tile_end / sk_iters_per_block +
            (tile_end % sk_iters_per_block != 0);
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
  if (plan.tiled_shape_m > INT32_MAX / plan.tiled_shape_n) {
    return StreamKSchedulePlan{};
  }
  plan.output_tile_count = plan.tiled_shape_m * plan.tiled_shape_n;
  if (plan.output_tile_count > INT32_MAX / plan.iters_per_tile) {
    return StreamKSchedulePlan{};
  }

  int full_waves = plan.output_tile_count / sm_count;
  int total_waves =
      full_waves + (plan.output_tile_count % sm_count != 0);
  int dp_blocks = 0;
  if (full_waves != total_waves &&
      plan.iters_per_tile > kMinStreamKIterations) {
    int dp_waves = full_waves > 1 ? full_waves - 1 : 0;
    dp_blocks = dp_waves * sm_count;
  }
  plan.dp_blocks = dp_blocks;
  plan.sk_tiles = plan.output_tile_count - plan.dp_blocks;
  if (plan.sk_tiles == 0) {
    return plan;
  }

  int requested_ctas =
      plan.sk_tiles > INT32_MAX / split_k_factor
          ? INT32_MAX
          : plan.sk_tiles * split_k_factor;
  int total_work = plan.sk_tiles * plan.iters_per_tile;
  int min_sized_ctas = total_work / kMinStreamKIterations;
  int target_ctas = min_sized_ctas < sm_count ? min_sized_ctas : sm_count;
  plan.sk_blocks = requested_ctas < target_ctas
                       ? requested_ctas
                       : target_ctas;
  if (plan.sk_blocks <= 0) {
    return StreamKSchedulePlan{};
  }

  plan.sk_iters_per_block =
      total_work / plan.sk_blocks +
      (total_work % plan.sk_blocks != 0);

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

template <int kStoreIterations, int kCtaN, int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_f16_from_f32_strided(
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

  if (k_tiles_to_compute < 2) {
    return;
  }

  tile::issue_cp_async_A4(smem->buffer[0].A, gA_base, tA_row, tA_col,
                                 stride_A);
  tile::issue_cp_async_B8(smem->buffer[0].B, gB_base, tB_row, tB_col,
                                 stride_B);
  cp_async::commit_group();
  --k_tiles_to_compute;

  tile::issue_cp_async_A4(smem->buffer[1].A, gA_base + kCtaK, tA_row,
                                 tA_col, stride_A);
  tile::issue_cp_async_B8(
      smem->buffer[1].B, gB_base + kCtaK * stride_B, tB_row, tB_col,
      stride_B);
  cp_async::commit_group();
  --k_tiles_to_compute;

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
    int b_ldsm_base = tile::offset_B(
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
    base::run_mma_tile_n256<K_PIPE_MAX - 2, true>(
        tCrC, tCrA, tCrB, smem, smem_read_offset, smem_write_offset, gA_next,
        gB_next, stride_A, stride_B, tA_row, tA_col, tB_row, tB_col,
        warp_m_id, warp_n_id, ldsmx4_row, ldsmx4_col, ldsmx4T_row,
        ldsmx4T_col);
  }

#pragma unroll
  for (; k_tiles_to_compute > -(K_PIPE_MAX - 1); --k_tiles_to_compute) {
    base::run_mma_tile_n256<K_PIPE_MAX - 2, false>(
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
      base::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][0][0], tCrC[m][n][0][0][1]);
      base::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 0 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][0][1][0], tCrC[m][n][0][1][1]);
      base::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][0][0], tCrC[m][n][1][0][1]);
      base::store_f32x2(
          sC, m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row,
          n * Tiled_MMA_N + warp_n_id * 8 + 1 * 32 + core_matrix_col * 2,
          kSmemStrideC, tCrC[m][n][1][1][0], tCrC[m][n][1][1][1]);
    }
  }

  __syncthreads();
  if (store_to_output) {
    store_gmem_f16_from_f32_strided<32, kCtaN, kStreamKThreads, kSmemStrideC>(
        reinterpret_cast<half *>(gOutput), sC, stride_B);
  } else {
    base::store_gmem_f32_strided<32, kCtaN, kStreamKThreads, kSmemStrideC>(
        reinterpret_cast<float *>(gOutput), sC, kCtaN);
  }
  __syncthreads();
}

template <int kStages>
__device__ __forceinline__ int process_one_segment(
    int work, int range_end, int tiled_shape_n,
    int iters_per_tile, int work_per_block, const half *A,
    const half *B, float *partials,
    half *C,
    HgemmSharedStorage<shape_mnk_n256, kStages> *smem,
    char *shared_memory, int N, int K, int tile_id_offset,
    int dp_blocks, int max_peers_per_tile, bool store_to_output) {
  if (work < 0 || work >= range_end || tiled_shape_n <= 0 ||
      iters_per_tile <= 0 || work_per_block <= 0) {
    return kInvalidWork;
  }

  int tile_id = work / iters_per_tile;
  int tile_work_begin = tile_id * iters_per_tile;
  int segment_work_end =
      min(range_end, tile_work_begin + iters_per_tile);
  int k_tile_begin = work - tile_work_begin;
  int k_tile_end = segment_work_end - tile_work_begin;
  if (k_tile_begin >= k_tile_end) {
    return kInvalidWork;
  }

  int local_tile_id = tile_id;
  int output_tile_id = tile_id_offset + local_tile_id;
  int tile_m = output_tile_id / tiled_shape_n;
  int tile_n = output_tile_id % tiled_shape_n;
  char *gOutput = nullptr;
  if (!store_to_output) {
    int first_peer_block =
        static_cast<int>(tile_work_begin / work_per_block);
    int peer_slot = static_cast<int>(blockIdx.x) - dp_blocks - first_peer_block;
    if (peer_slot < 0 || peer_slot >= max_peers_per_tile) {
      return kInvalidWork;
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
  int sk_iters_per_block = params.sk_iters_per_block;
  int max_peers_per_tile = params.max_peers_per_tile;

  if (iters_per_tile <= 0 || sk_tiles < 0 || dp_blocks < 0 ||
      dp_blocks > INT32_MAX / iters_per_tile ||
      (sk_tiles > 0 && sk_tiles > INT32_MAX / iters_per_tile)) {
    return;
  }

  extern __shared__ char shared_memory[];
  auto *smem = reinterpret_cast<
      HgemmSharedStorage<shape_mnk_n256, kStages> *>(shared_memory);

  bool store_to_output = static_cast<int>(blockIdx.x) < dp_blocks;
  int sk_block = static_cast<int>(blockIdx.x) - dp_blocks;
  if (!store_to_output &&
      (sk_block < 0 || sk_block >= sk_blocks || sk_iters_per_block <= 0)) {
    return;
  }

  int work_per_block = 0;
  int range_begin = 0;
  int range_end = 0;
  if (store_to_output) {
    work_per_block = iters_per_tile;
    range_begin = static_cast<int>(blockIdx.x) * iters_per_tile;
    range_end = range_begin + iters_per_tile;
  } else {
    int total_work = sk_tiles * iters_per_tile;
    work_per_block = sk_iters_per_block;
    range_begin = sk_block > total_work / work_per_block
                      ? total_work
                      : sk_block * work_per_block;
    int remaining_work = total_work - range_begin;
    range_end = remaining_work < work_per_block
                    ? total_work
                    : range_begin + work_per_block;
  }
  if (range_begin >= range_end) {
    return;
  }

  for (int work = range_begin; work < range_end;) {
    int next_work = process_one_segment<kStages>(
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
  int sk_iters_per_block = params.sk_iters_per_block;
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

} // namespace cuda_ops_core::detail::sm80::streamk
