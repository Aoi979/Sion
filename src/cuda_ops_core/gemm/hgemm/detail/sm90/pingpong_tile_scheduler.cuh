#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "persistent_tile_scheduler.cuh"

namespace cuda_ops_core::detail::sm90::scheduler {

class PingPongTileScheduler
    : public PersistentTileSchedulerBase<PingPongTileScheduler> {
  using Base = PersistentTileSchedulerBase<PingPongTileScheduler>;

public:
  static constexpr uint32_t kNumMmaWarpGroups = 2;

  using Base::Base;

  __device__ Tile initial_consumer_tile_impl(
      uint32_t consumer_warp_group_idx) {
    if (consumer_warp_group_idx == 1) {
      this->advance_to_next_work();
    }
    return this->current();
  }

  __device__ Tile next_consumer_tile_impl() {
    return this->next(kNumMmaWarpGroups);
  }

  __device__ bool is_last_consumer_tile_impl() const {
    return this->is_last_tile(kNumMmaWarpGroups);
  }
};

} // namespace cuda_ops_core::detail::sm90::scheduler
