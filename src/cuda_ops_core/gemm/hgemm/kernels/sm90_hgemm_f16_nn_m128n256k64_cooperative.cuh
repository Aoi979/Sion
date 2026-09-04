#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../detail/sm90/barrier.cuh"
#include "../detail/sm90/cluster.cuh"
#include "../detail/sm90/cooperative_tile_scheduler.cuh"
#include "../detail/sm90/pipeline.cuh"
#include "../detail/sm90/shared_memory.cuh"
#include "../detail/sm90/wgmma.cuh"
#include "../detail/sm90/tma.cuh"

using namespace ::cuda_ops_core::detail::sm90::barrier;
using namespace ::cuda_ops_core::detail::sm90::cluster;
using namespace ::cuda_ops_core::detail::sm90::pipeline;
using namespace ::cuda_ops_core::detail::sm90::shared_memory;
using namespace ::cuda_ops_core::detail::sm90::scheduler;
using namespace ::cuda_ops_core::detail::sm90::tma;
using namespace ::cuda_ops_core::detail::sm90::wgmma;

constexpr int kWarpSize = 32;
constexpr int kWarpGroupSize = 4 * kWarpSize;
constexpr int kCoopNumWarpGroups = 3;
constexpr int kCoopThreads = kCoopNumWarpGroups * kWarpGroupSize;
constexpr int kCoopStages = 4;
constexpr int kCoopClusterM = 1;
constexpr int kCoopClusterN = 2;
constexpr int kCoopCtaM = 128;
constexpr int kCoopCtaN = 256;
constexpr int kCoopCtaK = 64;
constexpr int kCoopGmmaK = 16;
constexpr int kCoopConsumers = 2;
constexpr int kCoopConsumerM = kCoopCtaM / kCoopConsumers;
constexpr int kCoopTmaBAtomN = 64;
constexpr int kCoopTmaBAtomK = 8;
constexpr int kCoopEpilogueTileN = 32;
constexpr int kCoopEpilogueStagesC = 4;
constexpr int kCoopEpilogueStagesD = 2;
constexpr int kCoopTmaStoreTileN = 64;
constexpr int kCoopTmaStoreStages =
    (kCoopEpilogueTileN * kCoopEpilogueStagesC) / kCoopTmaStoreTileN;
constexpr int kCoopSm90SharedCapacityBytes = 233472;
using Element = half;

static_assert(kCoopCtaN == 256);
static_assert(kCoopCtaK % kCoopGmmaK == 0);
static_assert(kCoopCtaN % kCoopTmaStoreTileN == 0);
static_assert(kCoopTmaStoreTileN % 64 == 0);
static_assert(kCoopTmaStoreStages >= 1);
static_assert(kCoopCtaM * kCoopEpilogueTileN * kCoopEpilogueStagesC >=
              kCoopCtaM * kCoopTmaStoreTileN * kCoopTmaStoreStages);


template <int CM, int CN>
__device__ __forceinline__ int cluster_rank_mn(int cm, int cn) {
  return cm + CM * cn;
}

template <int CM, int CN>
__device__ __forceinline__ uint16_t A_mcast_mask(int cm) {
  uint16_t mask = 0;
#pragma unroll
  for (int cn = 0; cn < CN; ++cn) {
    mask |= uint16_t(1u << cluster_rank_mn<CM, CN>(cm, cn));
  }
  return mask;
}

template <int CM, int CN>
__device__ __forceinline__ uint16_t B_mcast_mask(int cn) {
  uint16_t mask = 0;
#pragma unroll
  for (int cm = 0; cm < CM; ++cm) {
    mask |= uint16_t(1u << cluster_rank_mn<CM, CN>(cm, cn));
  }
  return mask;
}

__device__ __forceinline__ void consumer_warpgroups_sync() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kCoopConsumers * kWarpGroupSize)
               : "memory");
}

__device__ __forceinline__ int b_mn_smem_offset(int n_offset, int k_offset,
                                                int block_n) {
  return (n_offset % kCoopTmaBAtomN) +
         (n_offset / kCoopTmaBAtomN) * (kCoopTmaBAtomN * kCoopTmaBAtomK) +
         (k_offset / kCoopTmaBAtomK) * (block_n * kCoopTmaBAtomK);
}

template <int M, int K> struct ABuffer {
  half A[M * K];
};

template <int N, int K> struct BBuffer {
  half B[N * K];
};

template <int TileM, int TileN, int StagesC, int StagesD>
struct alignas(128) EpilogueBuffer {
  static_assert(StagesD <= StagesC);
  static_assert(TileN % 16 == 0);
  union {
    alignas(128) half C[TileM * TileN * StagesC];
    alignas(128) half D[TileM * TileN * StagesC];
  };
};

template <int M, int N, int K, int Stages> struct SharedStorage {
  static_assert(M == kCoopCtaM);
  static_assert(N % kCoopEpilogueTileN == 0);
  ABuffer<M, K> Abuffer[Stages];
  BBuffer<N, K> Bbuffer[Stages];
  EpilogueBuffer<kCoopCtaM, kCoopEpilogueTileN, kCoopEpilogueStagesC,
                 kCoopEpilogueStagesD>
      Epilogue;
};

template <typename SharedStorageT>
__device__ __forceinline__ void
store_accumulators_tma(SharedStorageT &smem, void const *tensor_map_c,
                       GemmTile tile, int consumer_id,
                       int warp_group_thread_idx,
                       float accumulator[kCoopCtaN / 16][8]) {
  int const row =
      (warp_group_thread_idx & 0xf) + (warp_group_thread_idx >> 5) * 16;
  int const row_in_cta = consumer_id * kCoopConsumerM + row;
  int const col_lane = ((warp_group_thread_idx >> 4) & 0x1) * 8;

#pragma unroll
  for (int n_store = 0; n_store < kCoopCtaN / kCoopTmaStoreTileN; ++n_store) {
    half *smem_d = &smem.Epilogue.D[(n_store % kCoopTmaStoreStages) *
                                    kCoopCtaM * kCoopTmaStoreTileN];
    uint32_t const smem_d_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_d));

#pragma unroll
    for (int inst_n = 0; inst_n < kCoopTmaStoreTileN / 16; ++inst_n) {
      alignas(16) half frag[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        frag[i] = __float2half_rn(
            accumulator[n_store * (kCoopTmaStoreTileN / 16) + inst_n][i]);
      }

      int const col = inst_n * 16 + col_lane;
      int const offset = row_in_cta * kCoopTmaStoreTileN + col;
      store_matrix_x4_m8n8(
          &smem_d[swizzled_half_offset_128b(smem_d_addr, offset)], frag);
    }

    fence_async_shared();
    consumer_warpgroups_sync();
    if (consumer_id == 0 && warp_group_thread_idx == 0) {
      tma_store(tensor_map_c, smem_d, tile.m * kCoopCtaM,
                tile.n * kCoopCtaN + n_store * kCoopTmaStoreTileN);
      tma_commit_group();
      tma_wait_group<0>();
    }
    consumer_warpgroups_sync();
  }
}

__global__ __launch_bounds__(kCoopThreads) void hgemm_cooperative_kernel(
    int M, int N, int K, const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    const __grid_constant__ CUtensorMap tensorMapC,
    PersistentTileSchedulerSm90Params scheduler_params) {
  constexpr int kGmemASliceSize = sizeof(Element) * kCoopCtaK * kCoopCtaM;
  constexpr int kGmemBSliceSize = sizeof(Element) * kCoopCtaK * kCoopCtaN;
  constexpr int kExpected_bytes = kGmemASliceSize + kGmemBSliceSize;

  int warp_group_id = threadIdx.x / kWarpGroupSize;
  int warp_id = threadIdx.x / kWarpSize;
  int const warp_group_thread_idx = threadIdx.x % kWarpGroupSize;
  int warp_in_wg = warp_id % 4;
  enum class WarpGroupRole { Producer, Consumer0, Consumer1 };
  WarpGroupRole role = warp_group_id == 0
                           ? WarpGroupRole::Producer
                           : (warp_group_id == 1 ? WarpGroupRole::Consumer0
                                                 : WarpGroupRole::Consumer1);
  int lane_id = threadIdx.x % kWarpSize;

  extern __shared__ __align__(128) char shared_memory[];
  using MainLoopSharedStorage =
      SharedStorage<kCoopCtaM, kCoopCtaN, kCoopCtaK, kCoopStages>;
  static_assert(sizeof(MainLoopSharedStorage) <= kCoopSm90SharedCapacityBytes);
  MainLoopSharedStorage *smem =
      reinterpret_cast<MainLoopSharedStorage *>(shared_memory);

  int const K_TILE_MAX = K / kCoopCtaK;
  CooperativeTileScheduler scheduler(scheduler_params);
  using MainloopPipeline = Pipeline<kCoopStages>;
  __shared__ MainloopPipeline::SharedStorage pipeline_storage;
  MainloopPipeline pipeline(
      pipeline_storage, kCoopClusterM + kCoopClusterN - 1);

  if (threadIdx.x == 0) {
    pipeline.initialize(
        kCoopConsumers * (kCoopClusterM + kCoopClusterN - 1));
  }
  __syncthreads();
  pipeline.fence_barrier_init();
  cluster_sync();

  if (role == WarpGroupRole::Producer) {
    warpgroup_reg_dealloc<40>();
    if (warp_in_wg == 0) {
      if (lane_id == 0) {
        PipelineState<kCoopStages> smem_pipe_write;
        auto cluster_rank = block_id_in_cluster();
        auto cluster_m_rank = cluster_rank.x;
        auto cluster_n_rank = cluster_rank.y;
        auto a_multicast_mask =
            A_mcast_mask<kCoopClusterM, kCoopClusterN>(cluster_m_rank);
        for (GemmTile tile = scheduler.current(); tile.valid;
             tile = scheduler.next_producer_tile()) {
          for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
            pipeline.producer_acquire(smem_pipe_write);
            pipeline.producer_expect_transaction(smem_pipe_write,
                                                 kExpected_bytes);
            uint64_t *full_barrier =
                pipeline.producer_get_barrier(smem_pipe_write);

            auto smem_a_rank_offset =
                cluster_n_rank * (kCoopCtaM / kCoopClusterN) * kCoopCtaK;
            // load A
            tma_multicast_load(
                &smem->Abuffer[smem_pipe_write.stage_idx].A[smem_a_rank_offset],
                &tensorMapA, full_barrier, a_multicast_mask,
                tile.m * kCoopCtaM +
                    cluster_n_rank * (kCoopCtaM / kCoopClusterN),
                k_tile * (kCoopCtaK / 64));

            // load B in GMMA MN-major SW128 order: each atom is N64 x K8.
#pragma unroll
            for (int k_atom = 0; k_atom < kCoopCtaK; k_atom += kCoopTmaBAtomK) {
              half *smem_b_atom =
                  &smem->Bbuffer[smem_pipe_write.stage_idx]
                       .B[b_mn_smem_offset(0, k_atom, kCoopCtaN)];
              tma_load(smem_b_atom, &tensorMapB, full_barrier,
                       k_tile * kCoopCtaK + k_atom,
                       (tile.n * kCoopCtaN) / kCoopTmaBAtomN);
            }

            pipeline.producer_commit(smem_pipe_write, kExpected_bytes);
            smem_pipe_write.advance();
          }
        }
      }
    }
  } else {
    const int consumer_id = role == WarpGroupRole::Consumer0 ? 0 : 1;
    warpgroup_reg_alloc<232>();
    float accumulator[kCoopCtaN / 16][8];
    PipelineState<kCoopStages> smem_pipe_read;
    PipelineState<kCoopStages> smem_pipe_release = smem_pipe_read;
    for (int i = 0; i < kCoopStages; i++) {
      pipeline.consumer_release(static_cast<uint32_t>(i),
                                static_cast<uint32_t>(warp_group_thread_idx));
    }
    for (GemmTile tile = scheduler.initial_consumer_tile(consumer_id);
         tile.valid; tile = scheduler.next_consumer_tile()) {

      pipeline.consumer_wait(smem_pipe_read);
      wgmma_fence();
      wgmma_m64n256k16_f32_f16_f16<0, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx].B[0]);
      wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK + 1 * kCoopGmmaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx]
               .B[b_mn_smem_offset(0, 1 * kCoopGmmaK, kCoopCtaN)]);
      wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK + 2 * kCoopGmmaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx]
               .B[b_mn_smem_offset(0, 2 * kCoopGmmaK, kCoopCtaN)]);
      wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK + 3 * kCoopGmmaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx]
               .B[b_mn_smem_offset(0, 3 * kCoopGmmaK, kCoopCtaN)]);
      wgmma_commit_group();
      smem_pipe_read.advance();

      for (int k_tile = 1; k_tile < K_TILE_MAX; k_tile++) {
        pipeline.consumer_wait(smem_pipe_read);
        wgmma_fence();
        wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx].B[0]);
        wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK + 1 * kCoopGmmaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx]
                 .B[b_mn_smem_offset(0, 1 * kCoopGmmaK, kCoopCtaN)]);
        wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK + 2 * kCoopGmmaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx]
                 .B[b_mn_smem_offset(0, 2 * kCoopGmmaK, kCoopCtaN)]);
        wgmma_m64n256k16_f32_f16_f16<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK + 3 * kCoopGmmaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx]
                 .B[b_mn_smem_offset(0, 3 * kCoopGmmaK, kCoopCtaN)]);

        wgmma_commit_group();
        wgmma_wait_group<1>();
        pipeline.consumer_release(
            smem_pipe_release,
            static_cast<uint32_t>(warp_group_thread_idx));
        smem_pipe_read.advance();
        smem_pipe_release.advance();
      }
      wgmma_wait_group<0>();
      pipeline.consumer_release(
          smem_pipe_release, static_cast<uint32_t>(warp_group_thread_idx));
      smem_pipe_release.advance();
      store_accumulators_tma(*smem, &tensorMapC, tile, consumer_id,
                             warp_group_thread_idx, accumulator);
    }
  }
}

inline cudaError_t map_cu_result(CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// IMPORTANT: B matrix is N-major
inline cudaError_t make_b_tensor_map(CUtensorMap *map, half const *ptr, int n,
                                     int k) {
  uint64_t shape[] = {64, static_cast<uint64_t>(k),
                      static_cast<uint64_t>(n / 64)};
  uint64_t stride[] = {static_cast<uint64_t>(sizeof(half)) *
                           static_cast<uint64_t>(n),
                       64ull * sizeof(half)};
  uint32_t box_shape[] = {64u, static_cast<uint32_t>(kCoopTmaBAtomK),
                          static_cast<uint32_t>(kCoopCtaN / 64)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<half *>(ptr), shape,
      stride, box_shape, box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

inline cudaError_t make_a_tensor_map(CUtensorMap *map, half const *ptr, int m,
                                     int k) {
  // IMPORTANT: Cluster shape is 1 x 2 x 1 (M N K)

  uint64_t shape[] = {64, static_cast<uint64_t>(m),
                      static_cast<uint64_t>(k / 64)};
  uint64_t stride[] = {static_cast<uint64_t>(sizeof(half)) *
                           static_cast<uint64_t>(k),
                       64ull * sizeof(half)};
  // IMPORTANT: CTA 0 load the fisrt 64 rows and CTA 1 load the next 64 rows!
  uint32_t box_shape[] = {64u, static_cast<uint32_t>(kCoopCtaM / kCoopClusterN),
                          static_cast<uint32_t>(kCoopCtaK / 64)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<Element *>(ptr),
      shape, stride, box_shape, box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

// IMPORTANT: C matrix is row-major.  Store one 128x64 chunk at a time.
inline cudaError_t make_c_tensor_map(CUtensorMap *map, half *ptr, int m,
                                     int n) {
  uint64_t shape[] = {64, static_cast<uint64_t>(m),
                      static_cast<uint64_t>(n / 64)};
  uint64_t stride[] = {static_cast<uint64_t>(sizeof(half)) *
                           static_cast<uint64_t>(n),
                       64ull * sizeof(half)};
  uint32_t box_shape[] = {64u, static_cast<uint32_t>(kCoopCtaM),
                          static_cast<uint32_t>(kCoopTmaStoreTileN / 64)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, ptr, shape, stride, box_shape,
      box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

namespace sm90_hgemm_cooperative {

inline cudaError_t launch_hgemm_128x256x64_cooperative(
    half const *A, half const *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0, int max_swizzle_size = 32,
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic) {
  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || N <= 0 ||
      K <= 0 || M % kCoopCtaM != 0 || N % (kCoopCtaN * kCoopClusterN) != 0 ||
      K % kCoopCtaK != 0) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }

  int cluster_launch = 0;
  err =
      cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device);
  if (err != cudaSuccess) {
    return err;
  }
  if (cluster_launch == 0) {
    return cudaErrorInvalidDeviceFunction;
  }

  int sm_count = 0;
  err =
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  if (err != cudaSuccess) {
    return err;
  }

  PersistentTileSchedulerSm90Params scheduler;
  scheduler.initialize(M, N, kCoopCtaM, kCoopCtaN, kCoopClusterM, kCoopClusterN,
                       max_swizzle_size, raster_order);
  if (scheduler.blocks_per_problem == 0) {
    return cudaErrorInvalidValue;
  }

  CUtensorMap map_a;
  CUtensorMap map_b;
  CUtensorMap map_c;
  err = make_a_tensor_map(&map_a, A, M, K);
  if (err != cudaSuccess) {
    return err;
  }
  err = make_b_tensor_map(&map_b, B, N, K);
  if (err != cudaSuccess) {
    return err;
  }
  err = make_c_tensor_map(&map_c, C, M, N);
  if (err != cudaSuccess) {
    return err;
  }

  using KernelSharedStorage =
      SharedStorage<kCoopCtaM, kCoopCtaN, kCoopCtaK, kCoopStages>;
  static_assert(sizeof(KernelSharedStorage) <= kCoopSm90SharedCapacityBytes);

  err = cudaFuncSetAttribute(hgemm_cooperative_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(KernelSharedStorage)));
  if (err != cudaSuccess) {
    return err;
  }
  err =
      cudaFuncSetAttribute(hgemm_cooperative_kernel,
                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_cooperative_kernel,
                             cudaFuncAttributeRequiredClusterWidth,
                             kCoopClusterM);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_cooperative_kernel,
                             cudaFuncAttributeRequiredClusterHeight,
                             kCoopClusterN);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_cooperative_kernel,
                             cudaFuncAttributeRequiredClusterDepth, 1);
  if (err != cudaSuccess) {
    return err;
  }

  dim3 grid =
      PersistentTileSchedulerSm90Params::get_grid_shape(scheduler, sm_count);
  hgemm_cooperative_kernel<<<grid, kCoopThreads, sizeof(KernelSharedStorage),
                             stream>>>(M, N, K, map_a, map_b, map_c, scheduler);
  return cudaGetLastError();
}

inline cudaError_t
launch_hgemm_128x128x64_cooperative(half const *A, half const *B, half *C,
                                    int M, int N, int K,
                                    cudaStream_t stream = 0) {
  return launch_hgemm_128x256x64_cooperative(A, B, C, M, N, K, stream);
}

} // namespace sm90_hgemm_cooperative
