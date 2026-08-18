// Reference:
// https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/utils.h
#include <cute/tensor.hpp>

namespace cuda_ops_core::flash_attn::utils {

template <typename T> __host__ __device__ __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T> struct MaxOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }
};

template <typename T> struct SumOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x + y;
  }
};

template <int THREADS> struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

template <> struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
// MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{}); // ((2, 2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<2>(l)));
};

} // namespace cuda_ops_core::flash_attn::utils