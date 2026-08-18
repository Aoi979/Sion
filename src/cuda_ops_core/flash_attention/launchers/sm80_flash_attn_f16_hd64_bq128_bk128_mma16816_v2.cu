#include "../detail/sm80_flash_attn_f16_mma16816_v2_launcher.cuh"
#include <cuda_ops_core/registry.hpp>

namespace cuda_ops_core {

Status sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  return detail::launch_sm80_flash_attn_f16_mma16816_v2<64>(
      Q, K, V, O, heads, batch_size, seq_len, stream);
}

} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2,
    cuda_ops_core::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2",
        cuda_ops_core::sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2_launch, true,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 60,
         .required_dynamic_smem_bytes =
             cuda_ops_core::detail::Sm80FlashAttnF16V2Traits<64>::kSmemBytes,
         .required_threads_per_block =
             cuda_ops_core::detail::Sm80FlashAttnF16V2Traits<64>::kThreads},
        {.head_dim = 64,
         .block_q = cuda_ops_core::detail::Sm80FlashAttnF16V2Traits<64>::kBlockQ,
         .block_k = cuda_ops_core::detail::Sm80FlashAttnF16V2Traits<64>::kBlockK,
         .seq_len_multiple =
             cuda_ops_core::detail::Sm80FlashAttnF16V2Traits<64>::kSeqLenMultiple}));
