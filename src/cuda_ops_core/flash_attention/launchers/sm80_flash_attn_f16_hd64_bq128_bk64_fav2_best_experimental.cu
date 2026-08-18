#include "../kernels/experimental/sm80_flash_attn_f16_hd64_best.cuh"
#include <cuda_ops_core/registry.hpp>

namespace cuda_ops_core {

Status
sm80_flash_attn_f16_hd64_bq128_bk64_fav2_best_experimental_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr ||
      batch_size != 2 || heads != 8 || seq_len != 4096) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "experimental SM80 FlashAttention D64 is locked to "
        "B=2,H=8,Sq=Sk=4096,D=64");
  }

  detail::sm80_flash_attn_f16_hd64_best::Fav2StaticKernelPtrs params{
      Q, K, V, O};
  cudaError_t err =
      detail::sm80_flash_attn_f16_hd64_best::fav2_sm80::
          launch_flash_attn_v2_static_b2_sq4096_sk4096_h8_d64(params, stream);
  if (err != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd64_bq128_bk64_fav2_best_experimental,
    cuda_ops_core::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd64_bq128_bk64_fav2_best_experimental",
        cuda_ops_core::sm80_flash_attn_f16_hd64_bq128_bk64_fav2_best_experimental_launch,
        false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 10,
         .required_dynamic_smem_bytes =
             cuda_ops_core::detail::sm80_flash_attn_f16_hd64_best::fav2_sm80::
                 FlashAttnV2LaunchConfig<64, 128, 64>::kSmemBytes,
         .required_threads_per_block =
             cuda_ops_core::detail::sm80_flash_attn_f16_hd64_best::fav2_sm80::
                 FlashAttnV2LaunchConfig<64, 128, 64>::kThreads},
        {.head_dim = 64,
         .block_q = 128,
         .block_k = 64,
         .seq_len_multiple = 4096}));
