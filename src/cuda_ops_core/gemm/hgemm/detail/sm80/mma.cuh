#pragma once

#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuda_ops_core::detail::sm80::common {

__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
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

} // namespace ldsm

namespace mma {

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

} // namespace mma

__device__ __forceinline__ uint32_t &as_u32(half &x) {
  return *reinterpret_cast<uint32_t *>(&x);
}

__device__ __forceinline__ uint32_t pack_f32x2_to_f16x2(float x, float y) {
  __half2 xy = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t *>(&xy);
}


namespace hgemm_smem {

__device__ __forceinline__ int offset_A(int m, int k) {
  int k_vec = (k >> 3) ^ (m & 7);
  return (m << 6) + (k_vec << 3) + (k & 7);
}

__device__ __forceinline__ int offset_B(int n, int k) {
  int n_vec = (n >> 3) ^ (k & 7);
  return (k << 7) + (n_vec << 3) + (n & 7);
}

} // namespace hgemm_smem

namespace hgemm_epilogue {

template <int kStoreIterations, int kCtaN, int kElementsPerAccess,
          int kThreads, int kSmemStrideC>
__device__ __forceinline__ void store_gmem_strided(half *gC, const half *sC,
                                                   int strideC) {
  constexpr int kVecsPerRow = kCtaN / kElementsPerAccess;
  constexpr int kRowsPerStep = kThreads / kVecsPerRow;

  int vec_row = threadIdx.x / kVecsPerRow;
  int vec_col = threadIdx.x % kVecsPerRow;
  half *d_ptr = gC + vec_row * strideC + vec_col * kElementsPerAccess;
  const half *s_ptr = sC + vec_row * kSmemStrideC +
                      vec_col * kElementsPerAccess;
  int d_step = kRowsPerStep * strideC;
  constexpr int s_step = kRowsPerStep * kSmemStrideC;

#pragma unroll
  for (int i = 0; i < kStoreIterations; ++i) {
    *reinterpret_cast<uint4 *>(d_ptr) =
        *reinterpret_cast<const uint4 *>(s_ptr);
    d_ptr += d_step;
    s_ptr += s_step;
  }
}

} // namespace hgemm_epilogue

} // namespace cuda_ops_core::detail::sm80::common

