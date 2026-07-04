#include "../detail/sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_launcher.cuh"
#include <felix/registry.hpp>

namespace felix {

FelixStatus sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  return detail::launch_sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d<64, 2,
                                                                     64>(
      Q, K, V, O, heads, batch_size, seq_len, stream);
}

} // namespace felix

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d,
    felix::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d",
        felix::sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d_launch, true,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 50,
         .required_dynamic_smem_bytes =
             felix::detail::kSm80FlashAttnF16Bq64Bk64OptInSmemBytes,
         .required_threads_per_block =
             felix::detail::kSm80FlashAttnF16Bq64Bk64Threads},
        {.head_dim = 64,
         .block_q = felix::detail::kSm80FlashAttnF16Bq64Bk64BlockQ,
         .block_k = felix::detail::kSm80FlashAttnF16Bq64Bk64BlockK,
         .seq_len_multiple = 64}));
