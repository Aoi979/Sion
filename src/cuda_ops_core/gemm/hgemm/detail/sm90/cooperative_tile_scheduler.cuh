#pragma once

#include <cstdint>

#include "persistent_tile_scheduler.cuh"

namespace cuda_ops_core::detail::sm90::scheduler {

class CooperativeTileScheduler
    : public PersistentTileSchedulerBase<CooperativeTileScheduler> {
  using Base = PersistentTileSchedulerBase<CooperativeTileScheduler>;

public:
  using Base::Base;

  __device__ Tile initial_consumer_tile_impl(uint32_t) {
    return this->current();
  }

  __device__ Tile next_consumer_tile_impl() { return this->next(); }

  __device__ bool is_last_consumer_tile_impl() const {
    return this->is_last_tile();
  }
};

} // namespace cuda_ops_core::detail::sm90::scheduler
