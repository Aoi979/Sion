#include "../detail/sm80_flash_attn_f16_mma16816_v2_launcher.cuh"
#include <felix/registry.hpp>

namespace felix {

FelixStatus sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  return detail::launch_sm80_flash_attn_f16_mma16816_v2<64>(
      Q, K, V, O, heads, batch_size, seq_len, stream);
}

} // namespace felix

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2,
    felix::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2",
        felix::sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2_launch, true,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 60,
         .required_dynamic_smem_bytes =
             felix::detail::Sm80FlashAttnF16V2Traits<64>::kSmemBytes,
         .required_threads_per_block =
             felix::detail::Sm80FlashAttnF16V2Traits<64>::kThreads},
        {.head_dim = 64,
         .block_q = felix::detail::Sm80FlashAttnF16V2Traits<64>::kBlockQ,
         .block_k = felix::detail::Sm80FlashAttnF16V2Traits<64>::kBlockK,
         .seq_len_multiple =
             felix::detail::Sm80FlashAttnF16V2Traits<64>::kSeqLenMultiple}));
