#include "../kernels/experimental/sm80_flash_attn_f16_hd128_best.cuh"
#include <felix/registry.hpp>

namespace felix {

FelixStatus
sm80_flash_attn_f16_hd128_bq128_bk64_fav2_best_experimental_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr ||
      batch_size != 1 || heads != 16 || seq_len != 4096) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "experimental SM80 FlashAttention D128 is locked to "
        "B=1,H=16,Sq=Sk=4096,D=128");
  }

  detail::sm80_flash_attn_f16_hd128_best::Fav2StaticKernelPtrs params{
      Q, K, V, O};
  cudaError_t err = detail::sm80_flash_attn_f16_hd128_best::fav2_sm80::
      launch_flash_attn_v2_static_b1_sq4096_sk4096_h16_d128(params, stream);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace felix

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd128_bq128_bk64_fav2_best_experimental,
    felix::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd128_bq128_bk64_fav2_best_experimental",
        felix::sm80_flash_attn_f16_hd128_bq128_bk64_fav2_best_experimental_launch,
        false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 10,
         .required_dynamic_smem_bytes =
             felix::detail::sm80_flash_attn_f16_hd128_best::fav2_sm80::
                 FlashAttnV2D128LaunchConfig::kSmemBytes,
         .required_threads_per_block =
             felix::detail::sm80_flash_attn_f16_hd128_best::fav2_sm80::
                 FlashAttnV2D128LaunchConfig::kThreads},
        {.head_dim = 128,
         .block_q = 128,
         .block_k = 64,
         .seq_len_multiple = 4096}));
