#pragma once

#include <cstdint>

namespace cuda_ops_core::detail::sm80_flash_attn_f16_hd64_best {
__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <typename H>
__device__ __forceinline__ uint32_t &as_u32_ref(H (&x)[2]) {
  static_assert(sizeof(H) == 2);
  static_assert(sizeof(x) == sizeof(uint32_t));

  return *reinterpret_cast<uint32_t *>(&x[0]);
}

template <typename H>
__device__ __forceinline__ const uint32_t &as_u32_ref(const H (&x)[2]) {
  static_assert(sizeof(H) == 2);
  static_assert(sizeof(x) == sizeof(uint32_t));

  return *reinterpret_cast<const uint32_t *>(&x[0]);
}

namespace cp_async {

enum class CacheMode {
  CA, // cache all: L1 + L2
  CG  // cache global: L2 only
};

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

template <int N> __device__ __forceinline__ void wait_group() {
  static_assert(N >= 0 && N <= 7, "cp.async.wait_group N must be in [0, 7]");
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <CacheMode Mode, int Bytes>
__device__ __forceinline__ void copy(void *smem_ptr, const void *gmem_ptr) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                "cp.async.ca supports 4, 8, 16 bytes; cp.async.cg supports "
                "only 16 bytes");

  if constexpr (Mode == CacheMode::CA) {
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
                 :
                 : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr), "n"(Bytes));
  } else {
    static_assert(Bytes == 16, "cp.async.cg only supports 16 bytes");

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr));
  }
}

template <int Bytes>
__device__ __forceinline__ void ca(void *smem_ptr, const void *gmem_ptr) {
  copy<CacheMode::CA, Bytes>(smem_ptr, gmem_ptr);
}

template <int Bytes>
__device__ __forceinline__ void cg(void *smem_ptr, const void *gmem_ptr) {
  copy<CacheMode::CG, Bytes>(smem_ptr, gmem_ptr);
}

} // namespace cp_async

namespace ldsm {

enum class Trans { No, Yes };

constexpr Trans T = Trans::Yes;
constexpr Trans N = Trans::No;

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x1(uint32_t &d0, const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
                 "{%0}, [%1];\n"
                 : "=r"(d0)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
                 "{%0}, [%1];\n"
                 : "=r"(d0)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x2(uint32_t &d0, uint32_t &d1,
                                   const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x4(uint32_t &v0v1, uint32_t &v2v3,
                                   uint32_t &v4v5, uint32_t &v6v7,
                                   const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0v1), "=r"(v2v3), "=r"(v4v5), "=r"(v6v7)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0v1), "=r"(v2v3), "=r"(v4v5), "=r"(v6v7)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No, typename H>
__device__ __forceinline__ void x4(H (&v0v1)[2], H (&v2v3)[2], H (&v4v5)[2],
                                   H (&v6v7)[2], const void *smem_ptr) {
  x4<kTrans>(as_u32_ref(v0v1), as_u32_ref(v2v3), as_u32_ref(v4v5),
             as_u32_ref(v6v7), smem_ptr);
}

} // namespace ldsm

namespace mma {

__device__ __forceinline__ void
m16n8k16_f32f16f16f32(float &d0, float &d1, float &d2, float &d3,

                     uint32_t const &a0, uint32_t const &a1,
                     uint32_t const &a2, uint32_t const &a3,

                     uint32_t const &b0, uint32_t const &b1,

                     float const &c0, float const &c1, float const &c2,
                     float const &c3) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%10, %11, %12, %13};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <typename HA, typename HB>
__device__ __forceinline__ void
m16n8k16_f32f16f16f32(float &d0, float &d1, float &d2, float &d3,

                     HA const (&a0)[2], HA const (&a1)[2],
                     HA const (&a2)[2], HA const (&a3)[2],

                     HB const (&b0)[2], HB const (&b1)[2],

                     float const &c0, float const &c1, float const &c2,
                     float const &c3) {
  m16n8k16_f32f16f16f32(
      d0, d1, d2, d3, as_u32_ref(a0), as_u32_ref(a1), as_u32_ref(a2),
      as_u32_ref(a3), as_u32_ref(b0), as_u32_ref(b1), c0, c1, c2, c3);
}

__device__ __forceinline__ void
m16n8k16_f16f16f16(uint32_t &d0, uint32_t &d1,

                   uint32_t const &a0, uint32_t const &a1, uint32_t const &a2,
                   uint32_t const &a3,

                   uint32_t const &b0, uint32_t const &b1,

                   uint32_t const &c0, uint32_t const &c1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "
               "{%2, %3, %4, %5}, "
               "{%6, %7}, "
               "{%8, %9};\n"
               : "=r"(d0), "=r"(d1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0),
                 "r"(c1));
}

__device__ __forceinline__ void
m16n8k16_f16f16f16_accum(uint32_t &c0, uint32_t &c1,

                         uint32_t const &a0, uint32_t const &a1,
                         uint32_t const &a2, uint32_t const &a3,

                         uint32_t const &b0, uint32_t const &b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "
               "{%2, %3, %4, %5}, "
               "{%6, %7}, "
               "{%0, %1};\n"
               : "+r"(c0), "+r"(c1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void
m16n8k16_f32f16f16f32_accum(float &c0, float &c1, float &c2, float &c3,

                            uint32_t const &a0, uint32_t const &a1,
                            uint32_t const &a2, uint32_t const &a3,

                            uint32_t const &b0, uint32_t const &b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <typename HA, typename HB>
__device__ __forceinline__ void
m16n8k16_f32f16f16f32_accum(float &c0, float &c1, float &c2, float &c3,

                            HA const (&a0)[2], HA const (&a1)[2],
                            HA const (&a2)[2], HA const (&a3)[2],

                            HB const (&b0)[2], HB const (&b1)[2]) {
  m16n8k16_f32f16f16f32_accum(c0, c1, c2, c3,

                              as_u32_ref(a0), as_u32_ref(a1), as_u32_ref(a2),
                              as_u32_ref(a3),

                              as_u32_ref(b0), as_u32_ref(b1));
}

} // namespace mma

} // namespace cuda_ops_core::detail::sm80_flash_attn_f16_hd64_best
