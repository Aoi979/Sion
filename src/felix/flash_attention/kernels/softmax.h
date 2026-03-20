// Reference:
// https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h
#include <cute/tensor.hpp>

namespace felix::flash_attn::utils {
using namespace cute;

template <int kNRows> struct Softmax {

  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;

  __forceinline__ __device__ Softmax() {};

  template <bool Is_first, bool Check_inf = false, typename Tensor0,
            typename Tensor1>
  __forceinline__ __device__ void
  softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    

  }
};
} // namespace felix::flash_attn::utils
