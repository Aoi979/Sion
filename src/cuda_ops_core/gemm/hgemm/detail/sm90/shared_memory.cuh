#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace cuda_ops_core::detail::sm90::shared_memory {

// Store four m8n8 matrix fragments from the registers of a warp group.
__device__ __forceinline__ void store_matrix_x4_m8n8(half *smem_ptr,
                                                     half src[8]) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t const *regs = reinterpret_cast<uint32_t const *>(src);
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], "
               "{%1, %2, %3, %4};\n" : : "r"(smem), "r"(regs[0]),
               "r"(regs[1]), "r"(regs[2]), "r"(regs[3]) : "memory");
}

// Convert a logical half offset to the offset selected by a 128B swizzle.
__device__ __forceinline__ int swizzled_half_offset_128b(uint32_t base_addr,
                                                          int half_offset) {
  uint32_t const byte_addr =
      base_addr + static_cast<uint32_t>(half_offset) * sizeof(half);
  uint32_t const swizzled_byte_addr = byte_addr ^ ((byte_addr & 0x380u) >> 3);
  return static_cast<int>((swizzled_byte_addr - base_addr) / sizeof(half));
}

} // namespace cuda_ops_core::detail::sm90::shared_memory
