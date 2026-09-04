#include <cstdint>
#include <cstdio>
#include <limits>

#include <cuda_runtime.h>

#include "gemm/hgemm/detail/sm90/cooperative_tile_scheduler.cuh"
#include "gemm/hgemm/detail/sm90/pingpong_tile_scheduler.cuh"

namespace scheduler =
    ::cuda_ops_core::detail::sm90::scheduler;

namespace {

using scheduler::RasterOrder;
using scheduler::RasterOrderOptions;

constexpr uint64_t kTraceSteps = 5;

template <typename Tile>
__device__ uint64_t encode_tile(Tile tile) {
  if (!tile.valid) {
    return ~uint64_t{0};
  }
  return (uint64_t{static_cast<uint32_t>(tile.m)} << 32) |
         static_cast<uint32_t>(tile.n);
}

template <typename TileScheduler, typename Params>
__global__ void trace_consumer(Params params, uint32_t consumer_warp_group_idx,
                               uint64_t *trace, uint8_t *last) {
  TileScheduler scheduler(params);
  auto tile = scheduler.initial_consumer_tile(consumer_warp_group_idx);
  uint64_t *row = trace + uint64_t{blockIdx.x} * kTraceSteps;
  uint8_t *last_row = last + uint64_t{blockIdx.x} * kTraceSteps;
  for (int step = 0; step < kTraceSteps; ++step) {
    row[step] = encode_tile(tile);
    last_row[step] = scheduler.is_last_consumer_tile() ? 1 : 0;
    if (!tile.valid) {
      break;
    }
    tile = scheduler.next_consumer_tile();
  }
}

template <typename TileScheduler, typename Params>
__global__ void trace_producer(Params params, uint64_t *trace, uint8_t *last) {
  TileScheduler scheduler(params);
  auto tile = scheduler.current();
  uint64_t *row = trace + uint64_t{blockIdx.x} * kTraceSteps;
  uint8_t *last_row = last + uint64_t{blockIdx.x} * kTraceSteps;
  for (int step = 0; step < kTraceSteps; ++step) {
    row[step] = encode_tile(tile);
    last_row[step] = scheduler.is_last_tile() ? 1 : 0;
    if (!tile.valid) {
      break;
    }
    tile = scheduler.next_producer_tile();
  }
}

struct LegacyState {
  uint32_t logical_tiles_m = 0;
  uint32_t logical_tiles_n = 0;
  uint32_t problem_blocks_m = 0;
  uint32_t problem_blocks_n = 0;
  uint32_t cluster_shape_m = 1;
  uint32_t cluster_shape_n = 1;
  uint64_t blocks_per_problem = 0;
  int32_t log_swizzle_size = 0;
  RasterOrder raster_order = RasterOrder::AlongN;
};

struct LegacyTile {
  int m = -1;
  int n = -1;
  bool valid = false;
  bool in_bounds = false;
};

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint32_t round_up(uint32_t value, uint32_t multiple) {
  return multiple == 0 ? value : ceil_div(value, multiple) * multiple;
}

int min_int(int lhs, int rhs) { return lhs < rhs ? lhs : rhs; }

int32_t legacy_log_swizzle(uint32_t ctas_m, uint32_t ctas_n,
                           int max_swizzle_size) {
  uint32_t min_cta_dim = ctas_m < ctas_n ? ctas_m : ctas_n;
  if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
    return 3;
  }
  if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
    return 2;
  }
  if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
    return 1;
  }
  return 0;
}

RasterOrder legacy_raster(uint32_t tiles_m, uint32_t tiles_n,
                          RasterOrderOptions option) {
  if (option == RasterOrderOptions::Heuristic) {
    return tiles_n > tiles_m ? RasterOrder::AlongM : RasterOrder::AlongN;
  }
  return option == RasterOrderOptions::AlongN ? RasterOrder::AlongN
                                              : RasterOrder::AlongM;
}

LegacyState legacy_initialize(uint32_t ctas_m, uint32_t ctas_n,
                              uint32_t cluster_m, uint32_t cluster_n,
                              int max_swizzle_size,
                              RasterOrderOptions raster_option) {
  LegacyState state;
  state.logical_tiles_m = ctas_m;
  state.logical_tiles_n = ctas_n;
  state.cluster_shape_m = cluster_m == 0 ? 1 : cluster_m;
  state.cluster_shape_n = cluster_n == 0 ? 1 : cluster_n;
  state.log_swizzle_size =
      legacy_log_swizzle(ctas_m, ctas_n, max_swizzle_size);
  uint32_t swizzle = 1u << state.log_swizzle_size;
  state.problem_blocks_m =
      round_up(ctas_m, swizzle * state.cluster_shape_m);
  state.problem_blocks_n =
      round_up(ctas_n, swizzle * state.cluster_shape_n);
  state.raster_order = legacy_raster(state.problem_blocks_m,
                                     state.problem_blocks_n, raster_option);
  state.blocks_per_problem =
      uint64_t{state.problem_blocks_m} * uint64_t{state.problem_blocks_n};
  return state;
}

LegacyTile legacy_tile(LegacyState const &state, uint64_t linear_idx,
                       uint32_t cta_m_in_cluster,
                       uint32_t cta_n_in_cluster) {
  if (linear_idx >= state.blocks_per_problem) {
    return {};
  }

  uint32_t cluster_major = state.raster_order == RasterOrder::AlongN
                               ? state.cluster_shape_n
                               : state.cluster_shape_m;
  uint32_t cluster_minor = state.raster_order == RasterOrder::AlongN
                               ? state.cluster_shape_m
                               : state.cluster_shape_n;
  uint32_t cluster_blocks_major =
      state.raster_order == RasterOrder::AlongN
          ? state.problem_blocks_n / state.cluster_shape_n
          : state.problem_blocks_m / state.cluster_shape_m;

  uint64_t blocks_per_grid_dim = linear_idx / cluster_minor;
  uint64_t cluster_id = blocks_per_grid_dim / cluster_major;
  uint64_t cluster_major_offset =
      blocks_per_grid_dim - cluster_id * cluster_major;
  uint64_t cluster_minor_offset = state.raster_order == RasterOrder::AlongN
                                      ? cta_m_in_cluster
                                      : cta_n_in_cluster;

  uint64_t swizzle = uint64_t{1} << state.log_swizzle_size;
  uint64_t offset = cluster_id & (swizzle - 1u);
  uint64_t extra = cluster_id >> state.log_swizzle_size;
  uint64_t cluster_idx_major =
      cluster_blocks_major == 0 ? 0 : extra % cluster_blocks_major;
  uint64_t minor_group =
      cluster_blocks_major == 0 ? 0 : extra / cluster_blocks_major;
  uint64_t cluster_idx_minor = minor_group * swizzle + offset;

  uint32_t minor_work_idx = static_cast<uint32_t>(
      cluster_idx_minor * cluster_minor + cluster_minor_offset);
  uint32_t major_work_idx = static_cast<uint32_t>(
      cluster_idx_major * cluster_major + cluster_major_offset);

  uint32_t tile_m = state.raster_order == RasterOrder::AlongN
                        ? minor_work_idx
                        : major_work_idx;
  uint32_t tile_n = state.raster_order == RasterOrder::AlongN
                        ? major_work_idx
                        : minor_work_idx;
  bool in_bounds = tile_m < state.logical_tiles_m &&
                   tile_n < state.logical_tiles_n;
  return {static_cast<int>(tile_m), static_cast<int>(tile_n), true,
          in_bounds};
}

dim3 legacy_grid(LegacyState const &state, int sm_count,
                 int max_active_clusters, bool truncate_by_problem_size) {
  if (state.blocks_per_problem == 0) {
    return dim3{0, 1, 1};
  }

  uint32_t cluster_m = state.cluster_shape_m;
  uint32_t cluster_n = state.cluster_shape_n;
  uint32_t cluster_size = cluster_m * cluster_n;
  int total = static_cast<int>(state.blocks_per_problem);

  dim3 grid = state.raster_order == RasterOrder::AlongN
                  ? dim3(cluster_m, 1, 1)
                  : dim3(1, cluster_n, 1);

  if (cluster_size == 1) {
    if (state.raster_order == RasterOrder::AlongN) {
      grid.y = truncate_by_problem_size ? min_int(sm_count, total) : sm_count;
    } else {
      grid.x = truncate_by_problem_size ? min_int(sm_count, total) : sm_count;
    }
    return grid;
  }

  auto max_cta_occupancy = [&](int max_sm_per_gpc) {
    if (sm_count <= 0 || max_sm_per_gpc <= 0) {
      return 0u;
    }
    int min_num_gpc = sm_count < max_sm_per_gpc
                          ? 1
                          : sm_count / max_sm_per_gpc;
    int max_cta_per_gpc =
        max_sm_per_gpc - max_sm_per_gpc % static_cast<int>(cluster_size);
    int cta_per_device = min_num_gpc * max_cta_per_gpc;
    int residual_gpc =
        sm_count < max_sm_per_gpc ? 0 : sm_count % max_sm_per_gpc;
    int residual_cta =
        residual_gpc - residual_gpc % static_cast<int>(cluster_size);
    cta_per_device += residual_cta;
    cta_per_device = sm_count < cta_per_device ? sm_count : cta_per_device;
    return static_cast<uint32_t>(cta_per_device);
  };

  if (max_active_clusters != 0 &&
      max_active_clusters * static_cast<int>(cluster_size) <= sm_count) {
    if (state.raster_order == RasterOrder::AlongN) {
      int active_ctas = max_active_clusters * static_cast<int>(cluster_n);
      int problem_ctas = total / static_cast<int>(cluster_m);
      grid.y = truncate_by_problem_size
                   ? min_int(active_ctas, problem_ctas)
                   : active_ctas;
    } else {
      int active_ctas = max_active_clusters * static_cast<int>(cluster_m);
      int problem_ctas = total / static_cast<int>(cluster_n);
      grid.x = truncate_by_problem_size
                   ? min_int(active_ctas, problem_ctas)
                   : active_ctas;
    }
  } else {
    uint32_t cta_per_device = max_cta_occupancy(18);
    if (state.raster_order == RasterOrder::AlongN) {
      int active_ctas = static_cast<int>(cta_per_device / cluster_m);
      int problem_ctas = total / static_cast<int>(cluster_m);
      grid.y = truncate_by_problem_size
                   ? min_int(active_ctas, problem_ctas)
                   : active_ctas;
    } else {
      int active_ctas = static_cast<int>(cta_per_device / cluster_n);
      int problem_ctas = total / static_cast<int>(cluster_n);
      grid.x = truncate_by_problem_size
                   ? min_int(active_ctas, problem_ctas)
                   : active_ctas;
    }
  }
  return grid;
}

bool check(bool condition, char const *message, int case_id) {
  if (!condition) {
    std::fprintf(stderr, "case %d: %s\n", case_id, message);
    return false;
  }
  return true;
}

bool run_case(int case_id, int M, int N, int cta_m, int cta_n,
              uint32_t cluster_m, uint32_t cluster_n, int max_swizzle_size,
              RasterOrderOptions raster_option) {
  uint32_t ctas_m = ceil_div(static_cast<uint32_t>(M), cta_m);
  uint32_t ctas_n = ceil_div(static_cast<uint32_t>(N), cta_n);
  LegacyState expected = legacy_initialize(
      ctas_m, ctas_n, cluster_m, cluster_n, max_swizzle_size, raster_option);

  scheduler::PersistentTileSchedulerSm90Params actual;
  actual.initialize(M, N, cta_m, cta_n, cluster_m, cluster_n,
                    max_swizzle_size, raster_option);

  bool ok = true;
  uint32_t expected_cluster_major =
      expected.raster_order == RasterOrder::AlongN
          ? expected.cluster_shape_n
          : expected.cluster_shape_m;
  uint32_t expected_cluster_minor =
      expected.raster_order == RasterOrder::AlongN
          ? expected.cluster_shape_m
          : expected.cluster_shape_n;
  uint32_t expected_cluster_blocks_major =
      expected.raster_order == RasterOrder::AlongN
          ? expected.problem_blocks_n / expected.cluster_shape_n
          : expected.problem_blocks_m / expected.cluster_shape_m;

  ok &= check(expected.problem_blocks_m == ctas_m &&
                  expected.problem_blocks_n == ctas_n,
              "baseline case unexpectedly needs padding", case_id);
  ok &= check(actual.cluster_shape_major == expected_cluster_major,
              "cluster_shape_major changed", case_id);
  ok &= check(actual.cluster_shape_minor == expected_cluster_minor,
              "cluster_shape_minor changed", case_id);
  ok &= check(actual.cluster_blocks_major == expected_cluster_blocks_major,
              "cluster_blocks_major changed", case_id);
  ok &= check(actual.blocks_per_problem == expected.blocks_per_problem,
              "blocks_per_problem changed", case_id);
  ok &= check(actual.log_swizzle_size() ==
                  static_cast<uint32_t>(expected.log_swizzle_size),
              "log_swizzle_size changed", case_id);
  ok &= check(actual.raster_order_along_n() ==
                  (expected.raster_order == RasterOrder::AlongN),
              "raster_order changed", case_id);

  uint64_t first_invalid = expected.blocks_per_problem;
  for (uint64_t linear_idx = 0; linear_idx < first_invalid + 11;
       ++linear_idx) {
    uint32_t cta_m_rank = expected.raster_order == RasterOrder::AlongN
                              ? linear_idx % expected_cluster_minor
                              : 0;
    uint32_t cta_n_rank = expected.raster_order == RasterOrder::AlongN
                              ? 0
                              : linear_idx % expected_cluster_minor;
    scheduler::GemmTile got = actual.tile_for_linear_idx(linear_idx);
    LegacyTile want =
        legacy_tile(expected, linear_idx, cta_m_rank, cta_n_rank);
    if (got.m != want.m || got.n != want.n || got.valid != want.valid) {
      std::fprintf(stderr,
                   "case %d: tile mismatch at linear=%llu rank=(%u,%u) "
                   "got=(%d,%d,%d) want=(%d,%d,%d)\n",
                   case_id, static_cast<unsigned long long>(linear_idx),
                   cta_m_rank, cta_n_rank, got.m, got.n, got.valid, want.m,
                   want.n, want.valid);
      ok = false;
      return ok;
    }
    if (want.valid && !want.in_bounds) {
      std::fprintf(stderr, "case %d: baseline tile is out of bounds\n", case_id);
      ok = false;
      return ok;
    }
  }

  for (int sm_count : {1, 8, 18, 20, 36, 80, 132}) {
    for (int max_active_clusters : {0, 1, 2, 4, 100}) {
      for (bool truncate : {false, true}) {
        dim3 got = scheduler::PersistentTileSchedulerSm90Params::get_grid_shape(
            actual, sm_count, max_active_clusters, truncate);
        dim3 want = legacy_grid(expected, sm_count, max_active_clusters,
                                truncate);
        if (got.x != want.x || got.y != want.y || got.z != want.z) {
          std::fprintf(stderr,
                       "case %d: grid mismatch sm=%d max_clusters=%d "
                       "truncate=%d got=(%u,%u,%u) want=(%u,%u,%u)\n",
                       case_id, sm_count, max_active_clusters, truncate,
                       got.x, got.y, got.z, want.x, want.y, want.z);
          ok = false;
        }
      }
    }
  }
  return ok;
}

template <typename TileScheduler, typename Params>
bool run_device_trace_case(Params params, bool pingpong, char const *name,
                           bool producer = false) {
  constexpr int kGridX = 3;
  constexpr uint64_t kInvalid = std::numeric_limits<uint64_t>::max();

  uint64_t *trace = nullptr;
  uint8_t *last = nullptr;
  cudaError_t err =
      cudaMallocManaged(&trace, 2 * kGridX * kTraceSteps * sizeof(uint64_t));
  if (err == cudaSuccess) {
    err = cudaMallocManaged(&last, 2 * kGridX * kTraceSteps * sizeof(uint8_t));
  }
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s trace allocation failed: %s\n", name,
                 cudaGetErrorString(err));
    cudaFree(trace);
    cudaFree(last);
    return false;
  }

  bool ok = true;
  uint32_t trace_count = producer ? 1 : 2;
  for (uint32_t consumer = 0; consumer < trace_count; ++consumer) {
    uint64_t *consumer_trace =
        trace + uint64_t{consumer} * kGridX * kTraceSteps;
    uint8_t *consumer_last =
        last + uint64_t{consumer} * kGridX * kTraceSteps;
    err = cudaMemset(consumer_trace, 0xff,
                     kGridX * kTraceSteps * sizeof(uint64_t));
    if (err == cudaSuccess) {
      err = cudaMemset(consumer_last, 0xff,
                       kGridX * kTraceSteps * sizeof(uint8_t));
    }
    if (err != cudaSuccess) {
      std::fprintf(stderr, "%s trace memset failed: %s\n", name,
                   cudaGetErrorString(err));
      ok = false;
      break;
    }

    if (producer) {
      trace_producer<TileScheduler><<<dim3(kGridX, 1, 1), 1>>>(
          params, consumer_trace, consumer_last);
    } else {
      trace_consumer<TileScheduler><<<dim3(kGridX, 1, 1), 1>>>(
          params, consumer, consumer_trace, consumer_last);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "%s trace kernel launch failed: %s\n", name,
                   cudaGetErrorString(err));
      ok = false;
      break;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "%s trace kernel execution failed: %s\n", name,
                   cudaGetErrorString(err));
      ok = false;
      break;
    }

    LegacyState expected = legacy_initialize(3, 4, 1, 1, 1,
                                              RasterOrderOptions::Heuristic);
    uint64_t consumer_offset =
        producer ? 0 : (pingpong ? uint64_t{consumer * kGridX} : 0);
    uint64_t consumer_stride = producer ? kGridX
                                       : (pingpong ? uint64_t{2 * kGridX}
                                                   : kGridX);
    for (int block = 0; block < kGridX; ++block) {
      uint64_t linear_idx = static_cast<uint64_t>(block) + consumer_offset;
      for (uint64_t step = 0; step < kTraceSteps; ++step) {
        LegacyTile expected_tile = legacy_tile(expected, linear_idx, 0, 0);
        uint64_t expected_value = expected_tile.valid
                                      ? (uint64_t{static_cast<uint32_t>(
                                             expected_tile.m)}
                                         << 32) |
                                            static_cast<uint32_t>(
                                                expected_tile.n)
                                      : kInvalid;
        uint64_t index = static_cast<uint64_t>(block) * kTraceSteps + step;
        uint64_t got = consumer_trace[index];
        bool expected_last =
            !legacy_tile(expected, linear_idx + consumer_stride, 0, 0).valid;
        if (got != expected_value || consumer_last[index] != expected_last) {
          std::fprintf(stderr,
                       "%s trace mismatch consumer=%u block=%d step=%llu "
                       "got=(%llu,%u) want=(%llu,%u)\n",
                       name, consumer, block,
                       static_cast<unsigned long long>(step),
                       static_cast<unsigned long long>(got),
                       static_cast<unsigned>(consumer_last[index]),
                       static_cast<unsigned long long>(expected_value),
                       static_cast<unsigned>(expected_last));
          ok = false;
        }
        if (!expected_tile.valid) {
          break;
        }
        linear_idx += consumer_stride;
      }
    }
  }

  cudaError_t free_err = cudaFree(trace);
  if (free_err != cudaSuccess) {
    std::fprintf(stderr, "%s trace cudaFree failed: %s\n", name,
                 cudaGetErrorString(free_err));
    ok = false;
  }
  free_err = cudaFree(last);
  if (free_err != cudaSuccess) {
    std::fprintf(stderr, "%s last-trace cudaFree failed: %s\n", name,
                 cudaGetErrorString(free_err));
    ok = false;
  }
  return ok;
}

bool run_device_consumer_traces() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaErrorNoDevice || device_count == 0) {
    cudaGetLastError();
    std::puts("sm90 scheduler: device traces skipped (no CUDA device)");
    return true;
  }
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                 cudaGetErrorString(err));
    return false;
  }

  scheduler::PersistentTileSchedulerSm90Params pingpong_params;
  pingpong_params.initialize_from_tile_counts(
      3, 4, 1, 1, 1, RasterOrderOptions::Heuristic);
  bool ok = run_device_trace_case<
      scheduler::PingPongTileScheduler>(pingpong_params, true, "pingpong");
  ok &= run_device_trace_case<scheduler::PingPongTileScheduler>(
      pingpong_params, true, "pingpong producer", true);

  scheduler::PersistentTileSchedulerSm90Params cooperative_params;
  cooperative_params.initialize_from_tile_counts(
      3, 4, 1, 1, 1, scheduler::RasterOrderOptions::Heuristic);
  ok &= run_device_trace_case<
      scheduler::CooperativeTileScheduler>(
      cooperative_params, false, "cooperative");
  ok &= run_device_trace_case<
      scheduler::CooperativeTileScheduler>(
      cooperative_params, false, "cooperative producer", true);
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  int case_id = 0;

  for (RasterOrderOptions raster : {RasterOrderOptions::Heuristic,
                                    RasterOrderOptions::AlongM,
                                    RasterOrderOptions::AlongN}) {
    for (int max_swizzle : {1, 2, 4, 8}) {
      // The production scheduler has an exact-tiling contract: M/N are
      // CTA- and cluster-aligned, so there are no padded work tiles.
      ok &= run_case(++case_id, 16 * 128, 16 * 128, 128, 128, 2, 1,
                     max_swizzle, raster);
      ok &= run_case(++case_id, 16 * 128, 16 * 128, 128, 128, 1, 2,
                     max_swizzle, raster);
      ok &= run_case(++case_id, 16 * 64, 16 * 96, 64, 96, 2, 2,
                     max_swizzle, raster);
    }
  }

  if (!ok) {
    return 1;
  }
  if (!run_device_consumer_traces()) {
    return 1;
  }
  std::puts("sm90 scheduler: exact-tiling behavior passed");
  return 0;
}
