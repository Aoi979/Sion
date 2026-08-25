#pragma once

#include <maca_fp16.h>
#include <__clang_maca_vector_types.h>
#include <stddef.h>

// 资料来源：MetaX Developer Documentation
// https://developer.metax-tech.com/doc

// Thin C++ wrappers for the MXC builtins.
// The MACA toolchain provides __NATIVE_VECTOR__ through its vector-types
// header. Keep this wrapper on that official spelling so it works with mxcc.

using v1u32 = __NATIVE_VECTOR__(1, unsigned);
using v2u32 = __NATIVE_VECTOR__(2, unsigned);
using v4u32 = __NATIVE_VECTOR__(4, unsigned);
using v4f16 = __NATIVE_VECTOR__(4, __fp16);
using v4f32 = __NATIVE_VECTOR__(4, float);

namespace mxc {

// The immediate-control operands of MXC memory/barrier builtins must remain
// compile-time constants. Encode them as non-type template parameters rather
// than ordinary inline-function arguments.

// Load 32 bits from global memory into shared memory.
// The return value is a synchronization flag, not loaded data.
template <int Offset, size_t Mask = 0, bool Ret0En = true,
          bool SaddrFlag = true, bool PredNeg = false, bool IsAsync = true>
__device__ __forceinline__ v1u32 ldg_b32_bsm(void* shared_addr,
                                              void* global_addr) {
  return __builtin_mxc_ldg_b32_bsm(shared_addr, global_addr, Offset, Mask,
                                   Ret0En, SaddrFlag, PredNeg, IsAsync);
}

// Load 64 bits from global memory into shared memory.
// The return value is a synchronization flag, not loaded data.
template <int Offset, size_t Mask = 0, bool Ret0En = true,
          bool SaddrFlag = true, bool PredNeg = false, bool IsAsync = true>
__device__ __forceinline__ v2u32 ldg_b64_bsm(void* shared_addr,
                                              void* global_addr) {
  return __builtin_mxc_ldg_b64_bsm(shared_addr, global_addr, Offset, Mask,
                                   Ret0En, SaddrFlag, PredNeg, IsAsync);
}

// Load 128 bits from global memory into shared memory.
// The return value is a synchronization flag, not loaded data.
template <int Offset, size_t Mask = 0, bool Ret0En = true,
          bool SaddrFlag = true, bool PredNeg = false, bool IsAsync = true>
__device__ __forceinline__ v4u32 ldg_b128_bsm(void* shared_addr,
                                               void* global_addr) {
  return __builtin_mxc_ldg_b128_bsm(shared_addr, global_addr, Offset, Mask,
                                    Ret0En, SaddrFlag, PredNeg, IsAsync);
}

// F16 16x16x16 matrix multiply-accumulate: D = A * B + C.
__device__ __forceinline__ v4f32 mma_16x16x16f16(v4f16 a, v4f16 b, v4f32 c) {
  return __builtin_mxc_mma_16x16x16f16(a, b, c);
}

// Wait until the selected outstanding global/shared-memory operations reach
// their target counts. flag == 0 waits for all preceding operations.
template <unsigned Flag>
__device__ __forceinline__ void arrive() {
  __builtin_mxc_arrive(Flag);
}

// Global-memory queue fence. gvmcnt is valid in [0, 63].
template <unsigned Gvmcnt>
__device__ __forceinline__ void arrive_gvmcnt() {
  __builtin_mxc_arrive_gvmcnt(Gvmcnt);
}

// Shared-memory queue fence. bsmcnt is valid in [0, 15].
template <unsigned Bsmcnt>
__device__ __forceinline__ void arrive_bsmcnt() {
  __builtin_mxc_arrive_bsmcnt(Bsmcnt);
}

// Shared-memory synchronization barrier with compiler-selected memory fence.
// This is a relatively relaxed barrier; prefer __syncthreads() when possible.
__device__ __forceinline__ void barrier() {
  __builtin_mxc_barrier();
}

// Instruction-level barrier. Prevents instruction reordering and waits for
// outstanding stores and instruction fetches to return.
__device__ __forceinline__ void barrier_inst() {
  __builtin_mxc_barrier_inst();
}

// Instruction barrier and optional memory fence.
template <int Flag>
__device__ __forceinline__ void barrier_ex() {
  __builtin_mxc_barrier_ex(Flag);
}

// Local wait paired with __builtin_mxc_ldg_b32_bsm.
template <int Scope>
__device__ __forceinline__ void barrier_and_wait1(int scope, v1u32 ret_flag) {
  __builtin_mxc_barrier_and_wait1(Scope, ret_flag);
}

// Local wait paired with __builtin_mxc_ldg_b64_bsm.
template <int Scope>
__device__ __forceinline__ void barrier_and_wait2(int scope, v2u32 ret_flag) {
  __builtin_mxc_barrier_and_wait2(Scope, ret_flag);
}

// Local wait paired with __builtin_mxc_ldg_b128_bsm.
template <int Scope>
__device__ __forceinline__ void barrier_and_wait4(int scope, v4u32 ret_flag) {
  __builtin_mxc_barrier_and_wait4(Scope, ret_flag);
}

// Small convenience wrappers. The builtin-compatible wrappers above remain
// the low-level interface; these helpers only name the commonly used values.
namespace helper {

static constexpr size_t all_lanes_mask = static_cast<size_t>(-1);

__device__ __forceinline__ void arrive_all() {
  mxc::arrive<0>();
}

__device__ __forceinline__ void barrier_all_memory() {
  mxc::barrier_ex<0>();
}

__device__ __forceinline__ void barrier_shared_memory() {
  mxc::barrier_ex<1>();
}

__device__ __forceinline__ void barrier_instruction_only() {
  mxc::barrier_ex<2>();
}

__device__ __forceinline__ void compiler_ordering_only() {
  mxc::barrier_ex<3>();
}

}  // namespace helper

}  // namespace mxc
