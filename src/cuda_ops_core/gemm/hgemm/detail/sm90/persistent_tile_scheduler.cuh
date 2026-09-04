#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "persistent_tile_scheduler_params.cuh"

namespace cuda_ops_core::detail::sm90::scheduler {

// Common persistent-grid traversal. The derived scheduler supplies only the
// consumer schedule; all policy dispatch is static.
template <typename Derived>
class PersistentTileSchedulerBase {
public:
  using Params = PersistentTileSchedulerSm90Params;
  using Tile = GemmTile;

  __device__ explicit PersistentTileSchedulerBase(Params const &params)
      : params_(params) {
    current_work_linear_idx_ =
        initial_work_linear_idx(params_.raster_order_along_n());
    total_grid_size_ =
        uint64_t{gridDim.x} * uint64_t{gridDim.y} * uint64_t{gridDim.z};
  }

  __device__ Tile current() const {
    return params_.tile_for_linear_idx(current_work_linear_idx_);
  }

  __device__ Tile initial_consumer_tile(uint32_t consumer_warp_group_idx) {
    return static_cast<Derived *>(this)->initial_consumer_tile_impl(
        consumer_warp_group_idx);
  }

  __device__ void advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t{advance_count};
  }

  __device__ Tile next(uint32_t advance_count = 1) {
    advance_to_next_work(advance_count);
    return current();
  }

  __device__ Tile next_producer_tile() { return next(); }

  __device__ Tile next_consumer_tile() {
    return static_cast<Derived *>(this)->next_consumer_tile_impl();
  }

  __device__ bool is_last_tile(uint32_t advance_count = 1) const {
    return !params_
                .tile_for_linear_idx(current_work_linear_idx_ +
                                     total_grid_size_ * uint64_t{advance_count})
                .valid;
  }

  __device__ bool is_last_consumer_tile() const {
    return static_cast<Derived const *>(this)->is_last_consumer_tile_impl();
  }

protected:
  __device__ Params const &params() const {
    return params_;
  }

  __device__ uint64_t initial_work_linear_idx(bool along_n) const {
    uint64_t block_minor =
        along_n ? uint64_t{blockIdx.x} : uint64_t{blockIdx.y};
    uint64_t block_major =
        along_n ? uint64_t{blockIdx.y} : uint64_t{blockIdx.x};
    uint64_t grid_minor = along_n ? uint64_t{gridDim.x} : uint64_t{gridDim.y};
    return block_minor + block_major * grid_minor;
  }

  Params params_{};
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 1;
};

inline int query_sm_count(int device = -1) {
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  return sm_count;
}

} // namespace cuda_ops_core::detail::sm90::scheduler
