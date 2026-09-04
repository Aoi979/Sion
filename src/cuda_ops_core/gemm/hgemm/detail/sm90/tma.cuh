#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace cuda_ops_core::detail::sm90::tma {

static constexpr uint64_t kCacheHintEvictLast = 0x14F0000000000000ull;

// Coordinates are the second and third coordinates of the tensor map. The
// first coordinate is fixed at zero for these row-major maps.
__device__ __forceinline__ void tma_load(half *dst, void const *tensor_map,
                                         uint64_t *bar, int coord_y,
                                         int coord_x) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];\n" : : "r"(dst_ptr),
               "l"(map_ptr), "r"(bar_ptr), "n"(0), "r"(coord_y),
               "r"(coord_x)
               : "memory");
}

__device__ __forceinline__ void
tma_multicast_load(half *dst, void const *tensor_map, uint64_t *bar,
                   uint16_t multicast_mask, int coord_y, int coord_x) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.multicast::cluster.L2::cache_hint"
               " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;\n" : : "r"(dst_ptr),
               "l"(map_ptr), "r"(bar_ptr), "h"(multicast_mask), "n"(0),
               "r"(coord_y), "r"(coord_x), "l"(kCacheHintEvictLast)
               : "memory");
}

__device__ __forceinline__ void tma_store(void const *tensor_map, half *src,
                                          int global_row, int global_col) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
               " [%0, {%2, %3, %4}], [%1];\n" : : "l"(map_ptr),
               "r"(src_ptr), "n"(0), "r"(global_row), "r"(global_col / 64)
               : "memory");
}

__device__ __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void tma_wait_group() {
  static_assert(PendingGroups >= 0 && PendingGroups <= 7);
  asm volatile("cp.async.bulk.wait_group.read %0;\n" : : "n"(PendingGroups)
               : "memory");
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

} // namespace cuda_ops_core::detail::sm90::tma
