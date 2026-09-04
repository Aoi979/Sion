#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "../detail/sm90/barrier.cuh"
#include "../detail/sm90/cluster.cuh"
#include "../detail/sm90/pipeline.cuh"
#include "../detail/sm90/shared_memory.cuh"
#include "../detail/sm90/wgmma.cuh"
#include "../detail/sm90/pingpong_tile_scheduler.cuh"
#include "../detail/sm90/tma.cuh"

namespace sm90_hgemm_pingpong {

using namespace ::cuda_ops_core::detail::sm90::barrier;
using namespace ::cuda_ops_core::detail::sm90::cluster;
using namespace ::cuda_ops_core::detail::sm90::pipeline;
using namespace ::cuda_ops_core::detail::sm90::shared_memory;
using namespace ::cuda_ops_core::detail::sm90::scheduler;
using namespace ::cuda_ops_core::detail::sm90::tma;
using namespace ::cuda_ops_core::detail::sm90::wgmma;

namespace detail {

using Element = half;

template <int Turns> class OrderedBarrier {
public:
  static_assert(Turns == 2,
                "OrderedBarrier currently supports two alternating turns");

  struct SharedStorage {
    uint64_t barriers[Turns];
  };

  struct State {
    uint32_t phase = 0;

    __device__ __forceinline__ void advance() { phase ^= 1u; }
  };

  __device__ explicit OrderedBarrier(SharedStorage &storage)
      : storage_(&storage) {}

  __device__ void initialize(uint32_t arrival_count = 1) {
#pragma unroll
    for (int i = 0; i < Turns; ++i) {
      ::cuda_ops_core::detail::sm90::barrier::init_barrier(
          &storage_->barriers[i], arrival_count);
    }
  }

  __device__ void wait(State state, uint32_t turn) {
    ::cuda_ops_core::detail::sm90::barrier::wait_barrier(
        &storage_->barriers[turn], state.phase);
  }

  __device__ void arrive(uint32_t turn, uint32_t count = 1) {
    ::cuda_ops_core::detail::sm90::barrier::arrive_barrier(
        &storage_->barriers[turn], count);
  }

  __device__ void arrive_next(uint32_t turn, uint32_t count = 1) {
    arrive((turn + 1u) % Turns, count);
  }

private:
  SharedStorage *storage_ = nullptr;
};

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kStages = 6;
constexpr int kWarpGroupSize = 128;
constexpr int kNumWarpGroups =
    1 + static_cast<int>(PingPongTileScheduler::kNumMmaWarpGroups);
constexpr int kThreads = kNumWarpGroups * kWarpGroupSize;
constexpr int kInstM = 64;
constexpr int kClusterM = 2;
constexpr int kClusterN = 1;
constexpr int kClusterK = 1;
constexpr int kTmaBAtomN = 64;
constexpr int kTmaBAtomK = 8;
constexpr int kTmaBAtomsPerRank = (kBlockK / kTmaBAtomK) / kClusterM;
static_assert(kBlockN % kClusterM == 0);
static_assert(kBlockN % kTmaBAtomN == 0);
static_assert(kBlockK % kTmaBAtomK == 0);
static_assert((kBlockK / kTmaBAtomK) % kClusterM == 0);
static_assert(kClusterN == 1);

struct SharedStorage {
  alignas(128) Element A[kBlockM * kBlockK * kStages];
  alignas(128) Element B[kBlockK * kBlockN * kStages];
  alignas(128) Element C[kBlockM * kBlockN];
};

inline cudaError_t map_cu_result(CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <int BlockMajor, int BlockMinor>
inline cudaError_t make_row_major_tensor_map(CUtensorMap *map,
                                             Element const *ptr,
                                             int height,
                                             int width) {
  static_assert(BlockMinor >= 64);
  static_assert(BlockMinor % 64 == 0);
  if (map == nullptr || ptr == nullptr || height <= 0 || width <= 0 ||
      width % 64 != 0) {
    return cudaErrorInvalidValue;
  }

  uint64_t shape[] = {64, static_cast<uint64_t>(height),
                      static_cast<uint64_t>(width / 64)};
  uint64_t stride[] = {static_cast<uint64_t>(sizeof(Element)) *
                           static_cast<uint64_t>(width),
                       64ull * sizeof(Element)};
  uint32_t box_shape[] = {64u, static_cast<uint32_t>(BlockMajor),
                          static_cast<uint32_t>(BlockMinor / 64)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
      const_cast<Element *>(ptr), shape, stride, box_shape, box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

template <int BlockN, int BlockK>
inline cudaError_t make_b_row_major_tensor_map(CUtensorMap *map,
                                               Element const *ptr,
                                               int n,
                                               int k) {
  static_assert(BlockN >= kTmaBAtomN);
  static_assert(BlockN % kTmaBAtomN == 0);
  static_assert(BlockK >= kTmaBAtomK);
  static_assert(BlockK % kTmaBAtomK == 0);
  if (map == nullptr || ptr == nullptr || n <= 0 || k <= 0 ||
      n % kTmaBAtomN != 0) {
    return cudaErrorInvalidValue;
  }

  uint64_t shape[] = {static_cast<uint64_t>(kTmaBAtomN),
                      static_cast<uint64_t>(k),
                      static_cast<uint64_t>(n / kTmaBAtomN)};
  uint64_t stride[] = {static_cast<uint64_t>(n) * sizeof(Element),
                       static_cast<uint64_t>(kTmaBAtomN) * sizeof(Element)};
  uint32_t box_shape[] = {static_cast<uint32_t>(kTmaBAtomN),
                          static_cast<uint32_t>(kTmaBAtomK),
                          static_cast<uint32_t>(BlockN / kTmaBAtomN)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
      const_cast<Element *>(ptr), shape, stride, box_shape, box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

__device__ __forceinline__ int b_mn_smem_offset(int n_offset, int k_offset) {
  return (n_offset / kTmaBAtomN) * (kTmaBAtomN * kTmaBAtomK) +
         (k_offset / kTmaBAtomK) * (kBlockN * kTmaBAtomK);
}


__device__ __forceinline__ uint16_t multicast_mask_b_cluster_m() {
  uint16_t mask = 0;
  uint32_t cta_n =
      ::cuda_ops_core::detail::sm90::cluster::block_id_in_cluster().y;
#pragma unroll
  for (uint32_t m = 0; m < kClusterM; ++m) {
    mask |= uint16_t{1u} << (m * kClusterN + cta_n);
  }
  return mask;
}

__device__ __forceinline__ void warpgroup_sync() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kWarpGroupSize) : "memory");
}

__device__ __forceinline__ void store_accumulators_tma(
    SharedStorage &smem, void const *tensor_map_c, GemmTile tile,
    int warp_group_thread_idx, float acc[2][8][8]) {
  int const row = (warp_group_thread_idx & 0xf) +
                  (warp_group_thread_idx >> 5) * 16;
  int const col_lane = ((warp_group_thread_idx >> 4) & 0x1) * 8;
  uint32_t const smem_c_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem.C));

#pragma unroll
  for (int mma_m = 0; mma_m < kBlockM / kInstM; ++mma_m) {
#pragma unroll
    for (int inst_n = 0; inst_n < kBlockN / 16; ++inst_n) {
      alignas(16) Element frag[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        frag[i] = __float2half_rn(acc[mma_m][inst_n][i]);
      }

      int const col = inst_n * 16 + col_lane;
      int const col_chunk = col / 64;
      int const col_in_chunk = col - col_chunk * 64;
      int const addr = mma_m * kInstM * kBlockN +
                       col_chunk * kInstM * 64 + row * 64 + col_in_chunk;
      store_matrix_x4_m8n8(
          &smem.C[swizzled_half_offset_128b(smem_c_addr, addr)], frag);
    }

    fence_async_shared();
    warpgroup_sync();
    if (warp_group_thread_idx == 0) {
      tma_store(tensor_map_c, &smem.C[mma_m * kInstM * kBlockN],
                tile.m * kBlockM + mma_m * kInstM, tile.n * kBlockN);
      tma_commit_group();
    }
  }
  tma_wait_group<0>();
}

__device__ __forceinline__ void clear_accumulators(float acc[2][8][8]) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[m][n][i] = 0.0f;
      }
    }
  }
}

__device__ __forceinline__ void keep_accumulators_live(float acc[2][8][8]) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        asm volatile("" : "+f"(acc[m][n][i]) :: "memory");
      }
    }
  }
}

template <bool ReleaseStages>
__device__ __forceinline__ void consume_k_tile(SharedStorage &smem,
                                               Pipeline<kStages> &pipeline,
                                               PipelineState<kStages> &read_pipe,
                                               PipelineState<kStages> &release_pipe,
                                               int warp_group_thread_idx,
                                               float acc[2][8][8]) {
  pipeline.consumer_wait(read_pipe);
  wgmma_fence();

#pragma unroll
  for (int mma_m = 0; mma_m < kBlockM / kInstM; ++mma_m) {
#pragma unroll
    for (int mma_k = 0; mma_k < kBlockK; mma_k += 16) {
      Element *sA = &smem.A[read_pipe.stage_idx * kBlockM * kBlockK +
                            mma_m * kInstM * kBlockK + mma_k];
      Element *sB =
          &smem.B[read_pipe.stage_idx * kBlockN * kBlockK + mma_k * kBlockN];
      wgmma_m64n128k16_f32_f16_f16<1, 1, 1, 0, 1>(acc[mma_m], sA, sB);
    }
  }

  wgmma_commit_group();
  if constexpr (ReleaseStages) {
    wgmma_wait_group<1>();
    pipeline.consumer_release(
        release_pipe, static_cast<uint32_t>(warp_group_thread_idx));
    release_pipe.advance();
  }
  read_pipe.advance();
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

} // namespace detail

__global__ __launch_bounds__(detail::kThreads) void hgemm_pingpong_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b,
    const __grid_constant__ CUtensorMap tensor_map_c,
    int M, int N, int K,
    PersistentTileSchedulerSm90Params scheduler_params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace detail;

  (void)M;
  (void)N;

  extern __shared__ __align__(128) uint8_t dynamic_smem[];
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(dynamic_smem);

  __shared__ Pipeline<kStages>::SharedStorage pipeline_storage;
  __shared__ OrderedBarrier<2>::SharedStorage math_turn_storage;
  __shared__ OrderedBarrier<2>::SharedStorage epilogue_turn_storage;

  Pipeline<kStages> pipeline(pipeline_storage, kClusterM + kClusterN - 1);
  OrderedBarrier<2> math_turn(math_turn_storage);
  OrderedBarrier<2> epilogue_turn(epilogue_turn_storage);

  int const thread_idx = static_cast<int>(threadIdx.x);
  int const warp_group_idx = thread_idx / kWarpGroupSize;
  int const warp_group_thread_idx = thread_idx % kWarpGroupSize;

  if (thread_idx == 0) {
    pipeline.initialize(kClusterM * kClusterN);
    math_turn.initialize();
    epilogue_turn.initialize();
  }
  __syncthreads();
  pipeline.fence_barrier_init();
  ::cuda_ops_core::detail::sm90::cluster::cluster_sync();

  int const k_tiles = K / kBlockK;
  PingPongTileScheduler scheduler(scheduler_params);

  enum class WarpGroupRole {
    Producer,
    Consumer0,
    Consumer1
  };
  WarpGroupRole role = warp_group_idx == 0
                           ? WarpGroupRole::Producer
                           : (warp_group_idx == 1 ? WarpGroupRole::Consumer0
                                                  : WarpGroupRole::Consumer1);

  if (role == WarpGroupRole::Producer) {
    warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      PipelineState<kStages> smem_pipe_write;
      uint32_t const cluster_rank =
          ::cuda_ops_core::detail::sm90::cluster::block_rank_in_cluster();
      uint32_t const cluster_m_rank =
          ::cuda_ops_core::detail::sm90::cluster::block_id_in_cluster().x;
      uint16_t const b_multicast_mask = multicast_mask_b_cluster_m();
      constexpr uint32_t load_bytes_a =
          sizeof(Element) * kBlockM * kBlockK;
      constexpr uint32_t load_bytes_b =
          sizeof(Element) * kBlockK * kBlockN;

      for (GemmTile tile = scheduler.current(); tile.valid;
           tile = scheduler.next_producer_tile()) {
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);
          bool const has_valid_b_tile = tile.valid;
          uint32_t expected_bytes = load_bytes_a;
          if (has_valid_b_tile &&
              (b_multicast_mask & (uint16_t{1u} << cluster_rank)) != 0) {
            expected_bytes += load_bytes_b;
          }

          pipeline.producer_expect_transaction(smem_pipe_write,
                                               expected_bytes);
          uint64_t *full_barrier =
              pipeline.producer_get_barrier(smem_pipe_write);
          tma_load(&smem.A[smem_pipe_write.stage_idx * kBlockM * kBlockK],
                   &tensor_map_a, full_barrier,
                   tile.m * kBlockM,
                   k_tile * kBlockK / 64);
          if (has_valid_b_tile && b_multicast_mask != 0) {
#pragma unroll
            for (int k_atom_iter = 0; k_atom_iter < kTmaBAtomsPerRank;
                 ++k_atom_iter) {
              int const k_atom =
                  (k_atom_iter * kClusterM +
                   static_cast<int>(cluster_m_rank)) *
                  kTmaBAtomK;
              Element *b_atom =
                  &smem.B[smem_pipe_write.stage_idx * kBlockN * kBlockK +
                          b_mn_smem_offset(0, k_atom)];
              tma_multicast_load(
                  b_atom, &tensor_map_b,
                  full_barrier, b_multicast_mask, k_tile * kBlockK + k_atom,
                  tile.n * kBlockN / kTmaBAtomN);
            }
          }
          pipeline.producer_commit(smem_pipe_write, expected_bytes);
          smem_pipe_write.advance();
        }
      }
    }
    return;
  }

  warpgroup_reg_alloc<232>();

  int const consumer_idx = role == WarpGroupRole::Consumer0 ? 0 : 1;
  OrderedBarrier<2>::State turn_state;
  PipelineState<kStages> smem_pipe_read;
  PipelineState<kStages> smem_pipe_release = smem_pipe_read;

  if (consumer_idx == 0) {
    for (int i = 0; i < kStages; ++i) {
      pipeline.consumer_release(static_cast<uint32_t>(i),
                                static_cast<uint32_t>(warp_group_thread_idx));
    }
  }

  if (consumer_idx == 1) {
    if (warp_group_thread_idx == 0) {
      math_turn.arrive(0);
      epilogue_turn.arrive(0);
    }
    smem_pipe_read.advance(k_tiles);
    smem_pipe_release.advance(k_tiles);
  }

  for (GemmTile tile = scheduler.initial_consumer_tile(consumer_idx);
       tile.valid; tile = scheduler.next_consumer_tile()) {
    math_turn.wait(turn_state, static_cast<uint32_t>(consumer_idx));

    float acc[2][8][8];
    clear_accumulators(acc);
    keep_accumulators_live(acc);

    consume_k_tile<false>(smem, pipeline, smem_pipe_read, smem_pipe_release,
                          warp_group_thread_idx, acc);
    for (int k_tile = 1; k_tile < k_tiles; ++k_tile) {
      consume_k_tile<true>(smem, pipeline, smem_pipe_read, smem_pipe_release,
                           warp_group_thread_idx, acc);
    }
    wgmma_wait_group<0>();
    pipeline.consumer_release(
        smem_pipe_release, static_cast<uint32_t>(warp_group_thread_idx));
    smem_pipe_release.advance();

    smem_pipe_read.advance(k_tiles);
    if (warp_group_thread_idx == 0) {
      math_turn.arrive_next(static_cast<uint32_t>(consumer_idx));
    }

    epilogue_turn.wait(turn_state, static_cast<uint32_t>(consumer_idx));

    store_accumulators_tma(smem, &tensor_map_c, tile, warp_group_thread_idx,
                           acc);
    if (warp_group_thread_idx == 0) {
      epilogue_turn.arrive_next(static_cast<uint32_t>(consumer_idx));
    }
    turn_state.advance();
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b;
  (void)tensor_map_c;
  (void)M;
  (void)N;
  (void)K;
  (void)scheduler_params;
#endif
}

inline cudaError_t launch_hgemm_128x128x64_pingpong(
    half const *A, half const *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0, int max_swizzle_size = 1,
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic) {
  using namespace detail;

  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || N <= 0 ||
      K <= 0 || M % (kBlockM * kClusterM) != 0 ||
      N % (kBlockN * kClusterN) != 0 ||
      K % kBlockK != 0) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }
  int cluster_launch = 0;
  err = cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch,
                               device);
  if (err != cudaSuccess) {
    return err;
  }
  if (cluster_launch == 0) {
    return cudaErrorInvalidDeviceFunction;
  }

  PersistentTileSchedulerSm90Params scheduler;
  scheduler.initialize(M, N, kBlockM, kBlockN, kClusterM, kClusterN,
                       max_swizzle_size, raster_order);

  CUtensorMap map_a;
  CUtensorMap map_b;
  CUtensorMap map_c;
  err = make_row_major_tensor_map<kBlockM, kBlockK>(&map_a, A, M, K);
  if (err != cudaSuccess) {
    return err;
  }
  err = make_b_row_major_tensor_map<kBlockN, kBlockK>(&map_b, B, N, K);
  if (err != cudaSuccess) {
    return err;
  }
  err = make_row_major_tensor_map<kInstM, kBlockN>(&map_c, C, M, N);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaFuncSetAttribute(
      hgemm_pingpong_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(SharedStorage)));
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributeRequiredClusterWidth,
                             kClusterM);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributeRequiredClusterHeight,
                             kClusterN);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributeRequiredClusterDepth,
                             kClusterK);
  if (err != cudaSuccess) {
    return err;
  }

  dim3 grid = PersistentTileSchedulerSm90Params::get_grid_shape(
      scheduler, query_sm_count());
  if (grid.x == 0 || grid.y == 0 || grid.z == 0 ||
      grid.x % kClusterM != 0 || grid.y % kClusterN != 0 ||
      grid.z % kClusterK != 0) {
    return cudaErrorInvalidConfiguration;
  }

  cudaLaunchAttribute launch_attr[1] = {};
  launch_attr[0].id = cudaLaunchAttributeClusterDimension;
  launch_attr[0].val.clusterDim.x = kClusterM;
  launch_attr[0].val.clusterDim.y = kClusterN;
  launch_attr[0].val.clusterDim.z = kClusterK;

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = sizeof(SharedStorage);
  config.stream = stream;
  config.attrs = launch_attr;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(&config, hgemm_pingpong_kernel, map_a, map_b,
                            map_c, M, N, K, scheduler);
}

} // namespace sm90_hgemm_pingpong
