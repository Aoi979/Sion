#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "barrier.cuh"

namespace cuda_ops_core::detail::sm90::pipeline {

// Cursor for a phase-bit ring of shared-memory pipeline stages.
template <int Stages> struct PipelineState {
  static_assert(Stages > 0);

  int phase = 0;
  int stage_idx = 0;

  __device__ __forceinline__ void advance(int steps = 1) {
    int const next_stage = stage_idx + steps;
    phase ^= (next_stage / Stages) & 1;
    stage_idx = next_stage % Stages;
  }
};

// TMA load pipeline over a ring of full and empty cluster barriers.
//
// The pipeline owns no shared-memory storage. SharedStorage is a layout type;
// the caller allocates an instance in __shared__ memory and passes it to the
// pipeline object.
template <int Stages> class Pipeline {
public:
  static_assert(Stages > 0);

  using State = PipelineState<Stages>;

  struct SharedStorage {
    uint64_t full[Stages];
    uint64_t empty[Stages];
  };

  __device__ explicit Pipeline(SharedStorage &storage,
                               uint32_t signaling_thread_count)
      : storage_(&storage),
        signaling_thread_count_(signaling_thread_count) {}

  // Called by the initializing thread before the cluster barrier-init fence.
  __device__ void initialize(uint32_t empty_arrival_count,
                             uint32_t full_arrival_count = 1) {
#pragma unroll
    for (int i = 0; i < Stages; ++i) {
      barrier::init_barrier(&storage_->full[i], full_arrival_count);
      barrier::init_barrier(&storage_->empty[i], empty_arrival_count);
    }
  }

  // Called by all threads after the initializing thread has initialized the
  // barrier array and the CTA has synchronized.
  __device__ void fence_barrier_init() {
    barrier::fence_barrier_init();
  }

  __device__ void producer_acquire(State state) {
    barrier::wait_barrier(&storage_->empty[state.stage_idx], state.phase);
  }

  __device__ void producer_expect_transaction(State state,
                                              uint32_t transaction_bytes) {
    barrier::expect_tma_bytes(&storage_->full[state.stage_idx],
                              transaction_bytes);
  }

  // The TMA instruction completes a non-empty full barrier transaction
  // itself. Preserve the software completion path for an empty transaction.
  __device__ void producer_commit(State state, uint32_t transaction_bytes) {
    if (transaction_bytes == 0) {
      barrier::arrive_barrier(&storage_->full[state.stage_idx]);
    }
  }

  __device__ uint64_t *producer_get_barrier(State state) {
    return &storage_->full[state.stage_idx];
  }

  __device__ void consumer_wait(State state) {
    barrier::wait_barrier(&storage_->full[state.stage_idx], state.phase);
  }

  // Each signaling thread supplies one destination CTA rank. Non-signaling
  // threads call this too, but become no-ops based on the configured count.
  __device__ void consumer_release(State state, uint32_t rank_id) {
    consumer_release(static_cast<uint32_t>(state.stage_idx), rank_id);
  }

  __device__ void consumer_release(uint32_t stage, uint32_t rank_id) {
    if (rank_id < signaling_thread_count_) {
      barrier::arrive_barrier_remote(&storage_->empty[stage], rank_id);
    }
  }

private:
  SharedStorage *storage_ = nullptr;
  uint32_t signaling_thread_count_ = 0;
};

} // namespace cuda_ops_core::detail::sm90::pipeline
