#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "sm90_cluster.cuh"
#include "sm90_gemmPersistentTileScheduler.cuh"
#include "sm90_mbarrier.cuh"

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

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
  return (x & 0x3ffff) >> 4;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(Element *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode(16) << 16;
  desc |= matrix_descriptor_encode(1024) << 32;
  desc |= 1ull << 62;
  return desc;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_b(Element *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
  desc |= matrix_descriptor_encode(1024) << 16;
  desc |= matrix_descriptor_encode(4096) << 32;
  desc |= 1ull << 62;
  return desc;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void wgmma_wait_group() {
  static_assert(PendingGroups >= 0 && PendingGroups <= 7);
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(PendingGroups)
               : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma256(float d[kCoopCtaN / 16][8], half *sA,
                                         half *sB) {
  uint64_t desc_a = make_smem_desc_k_major(sA);
  uint64_t desc_b = make_smem_desc_mn_major_b(sB);
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %130, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
               " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
               " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
               " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
               " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
               " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
               " %104, %105, %106, %107, %108, %109, %110, %111,  "
               " %112, %113, %114, %115, %116, %117, %118, %119,  "
               " %120, %121, %122, %123, %124, %125, %126, %127},"
               " %128,"
               " %129,"
               " p,    %131,  %132,  %133,  %134;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                 "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                 "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                 "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                 "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                 "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                 "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
                 "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
                 "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
                 "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
                 "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
                 "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
                 "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
                 "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
                 "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
                 "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
                 "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
                 "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
                 "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
                 "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
                 "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
                 "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
                 "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
                 "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
                 "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
                 "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
                 "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
                 "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
                 "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
                 "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
                 "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <uint32_t RegCount> __device__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> __device__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ __forceinline__ void arrive_cluster_empty_barrier(uint64_t *bar,
                                                             uint32_t rank_id) {
  arrive_barrier_remote(bar, rank_id);
}

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

template <int Stages> struct PipelineState {
  int phase = 0;
  int stage_idx = 0;
  __device__ void advance() {
    stage_idx++;
    if (stage_idx == Stages) {
      phase ^= 1;
      stage_idx = 0;
    }
  }
};

__device__ __forceinline__ void tma_load(half *dst, void const *tensor_map,
                                         uint64_t *bar, int major_blk,
                                         int minor_offset) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];\n" ::"r"(dst_ptr),
               "l"(map_ptr), "r"(bar_ptr), "n"(0), "r"(minor_offset),
               "r"(major_blk)
               : "memory");
}

__device__ __forceinline__ void
tma_multicast_load(half *dst, void const *tensor_map, uint64_t *bar,
                   uint16_t mask, int major_blk, int minor_offset) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.multicast::cluster.L2::cache_hint"
               " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;\n" ::"r"(dst_ptr),
               "l"(map_ptr), "r"(bar_ptr), "h"(mask), "n"(0), "r"(minor_offset),
               "r"(major_blk), "l"(0x14F0000000000000ull)
               : "memory");
}

__device__ __forceinline__ void tma_store(void const *tensor_map, half *src,
                                          int global_row, int global_col) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
               " [%0, {%2, %3, %4}], [%1];\n" ::"l"(map_ptr),
               "r"(src_ptr), "n"(0), "r"(global_row), "r"(global_col / 64)
               : "memory");
}

__device__ __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int PendingGroups> __device__ __forceinline__ void tma_wait_group() {
  static_assert(PendingGroups >= 0 && PendingGroups <= 7);
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(PendingGroups)
               : "memory");
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void consumer_warpgroups_sync() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kCoopConsumers * kWarpGroupSize)
               : "memory");
}

__device__ __forceinline__ void stmatrix(half *smem_ptr, half src[8]) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t const *regs = reinterpret_cast<uint32_t const *>(src);
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], "
               "{%1, %2, %3, %4};\n" ::"r"(smem),
               "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3])
               : "memory");
}

__device__ __forceinline__ int swizzle_128b_half_offset(uint32_t base_addr,
                                                        int half_offset) {
  uint32_t const byte_addr =
      base_addr + static_cast<uint32_t>(half_offset) * sizeof(half);
  uint32_t const swizzled_byte_addr = byte_addr ^ ((byte_addr & 0x380u) >> 3);
  return static_cast<int>((swizzled_byte_addr - base_addr) / sizeof(half));
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
      stmatrix(&smem_d[swizzle_128b_half_offset(smem_d_addr, offset)], frag);
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

// IMPORTANT: A matrix is K-major, B matrix is N-major
// cluster shape is 1x2(M x N), so 2 CTAs in a cluster need same A matrix, TMA
// CTA has 2 consumers, each one compute 64x256 tile so that complete the CTA
// tile(128x256)
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
  GemmPersistentTileScheduler scheduler(scheduler_params);
  __shared__ uint64_t full[kCoopStages];
  __shared__ uint64_t empty[kCoopStages];

  if (threadIdx.x == 0) {
    for (int i = 0; i < kCoopStages; i++) {
      init_barrier(&full[i], 1);
      init_barrier(&empty[i],
                   kCoopConsumers * (kCoopClusterM + kCoopClusterN - 1));
    }
  }
  __syncthreads();
  fence_barrier_init();
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
            wait_barrier(&empty[smem_pipe_write.stage_idx],
                         smem_pipe_write.phase);
            expect_tma_bytes(&full[smem_pipe_write.stage_idx], kExpected_bytes);

            auto smem_a_rank_offset =
                cluster_n_rank * (kCoopCtaM / kCoopClusterN) * kCoopCtaK;
            // load A
            tma_multicast_load(
                &smem->Abuffer[smem_pipe_write.stage_idx].A[smem_a_rank_offset],
                &tensorMapA, &full[smem_pipe_write.stage_idx], a_multicast_mask,
                k_tile * (kCoopCtaK / 64),
                tile.m * kCoopCtaM +
                    cluster_n_rank * (kCoopCtaM / kCoopClusterN));

            // load B in GMMA MN-major SW128 order: each atom is N64 x K8.
#pragma unroll
            for (int k_atom = 0; k_atom < kCoopCtaK; k_atom += kCoopTmaBAtomK) {
              half *smem_b_atom =
                  &smem->Bbuffer[smem_pipe_write.stage_idx]
                       .B[b_mn_smem_offset(0, k_atom, kCoopCtaN)];
              tma_load(smem_b_atom, &tensorMapB,
                       &full[smem_pipe_write.stage_idx],
                       (tile.n * kCoopCtaN) / kCoopTmaBAtomN,
                       k_tile * kCoopCtaK + k_atom);
            }

            smem_pipe_write.advance();
          }
        }
      }
    }
  } else {
    const int consumer_id = role == WarpGroupRole::Consumer0 ? 0 : 1;
    warpgroup_reg_alloc<232>();
    // 8 is used because it is the number of registers each thread needs to
    // support Core Matrix x4
    float accumulator[kCoopCtaN / 16][8];
    PipelineState<kCoopStages> smem_pipe_read;
    PipelineState<kCoopStages> smem_pipe_release = smem_pipe_read;
    for (int i = 0; i < kCoopStages; i++) {
      if (warp_group_thread_idx < (kCoopClusterM + kCoopClusterN - 1)) {
        arrive_cluster_empty_barrier(&empty[i], warp_group_thread_idx);
      }
    }
    for (GemmTile tile = scheduler.initial_consumer_tile(consumer_id);
         tile.valid; tile = scheduler.next_consumer_tile()) {

      wait_barrier(&full[smem_pipe_read.stage_idx], smem_pipe_read.phase);
      wgmma_fence();
      wgmma256<0, 1, 1, 0, 1>(accumulator,
                              &smem->Abuffer[smem_pipe_read.stage_idx]
                                   .A[consumer_id * kCoopConsumerM * kCoopCtaK],
                              &smem->Bbuffer[smem_pipe_read.stage_idx].B[0]);
      wgmma256<1, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK + 1 * kCoopGmmaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx]
               .B[b_mn_smem_offset(0, 1 * kCoopGmmaK, kCoopCtaN)]);
      wgmma256<1, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK + 2 * kCoopGmmaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx]
               .B[b_mn_smem_offset(0, 2 * kCoopGmmaK, kCoopCtaN)]);
      wgmma256<1, 1, 1, 0, 1>(
          accumulator,
          &smem->Abuffer[smem_pipe_read.stage_idx]
               .A[consumer_id * kCoopConsumerM * kCoopCtaK + 3 * kCoopGmmaK],
          &smem->Bbuffer[smem_pipe_read.stage_idx]
               .B[b_mn_smem_offset(0, 3 * kCoopGmmaK, kCoopCtaN)]);
      wgmma_commit_group();
      smem_pipe_read.advance();

      for (int k_tile = 1; k_tile < K_TILE_MAX; k_tile++) {
        wait_barrier(&full[smem_pipe_read.stage_idx], smem_pipe_read.phase);
        wgmma_fence();
        wgmma256<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx].B[0]);
        wgmma256<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK + 1 * kCoopGmmaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx]
                 .B[b_mn_smem_offset(0, 1 * kCoopGmmaK, kCoopCtaN)]);
        wgmma256<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK + 2 * kCoopGmmaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx]
                 .B[b_mn_smem_offset(0, 2 * kCoopGmmaK, kCoopCtaN)]);
        wgmma256<1, 1, 1, 0, 1>(
            accumulator,
            &smem->Abuffer[smem_pipe_read.stage_idx]
                 .A[consumer_id * kCoopConsumerM * kCoopCtaK + 3 * kCoopGmmaK],
            &smem->Bbuffer[smem_pipe_read.stage_idx]
                 .B[b_mn_smem_offset(0, 3 * kCoopGmmaK, kCoopCtaN)]);

        wgmma_commit_group();
        wgmma_wait_group<1>();
        if (warp_group_thread_idx < (kCoopClusterM + kCoopClusterN - 1)) {
          arrive_cluster_empty_barrier(&empty[smem_pipe_release.stage_idx],
                                       warp_group_thread_idx);
        }
        smem_pipe_read.advance();
        smem_pipe_release.advance();
      }
      wgmma_wait_group<0>();
      if (warp_group_thread_idx < (kCoopClusterM + kCoopClusterN - 1)) {
        arrive_cluster_empty_barrier(&empty[smem_pipe_release.stage_idx],
                                     warp_group_thread_idx);
        smem_pipe_release.advance();
      }
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

// IMPORTANT: Enable TMA multicast!!!!!
// A matrix is K-major
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
