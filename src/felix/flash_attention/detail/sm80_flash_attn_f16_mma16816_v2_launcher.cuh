#pragma once

#include "../kernels/sm80_flash_attn_f16_mma16816_v2.cuh"
#include <felix/status.hpp>

#include <cmath>

namespace felix::detail {

template <int HeadDim> struct Sm80FlashAttnF16V2Traits {
  using Config = sm80_flash_attn_v2::fav2_sm80::FlashAttnV2Sm80Config<HeadDim>;

  static_assert(Config::kSupported,
                "SM80 FlashAttention v2 supports this head dimension");

  static constexpr uint32_t kBlockQ = Config::kBlockMValue;
  static constexpr uint32_t kBlockK = Config::kBlockNValue;
  static constexpr uint32_t kThreads = Config::kThreads;
  static constexpr uint32_t kSmemBytes = Config::kSmemBytes;
  static constexpr uint32_t kSeqLenMultiple =
      (kBlockQ % kBlockK == 0) ? kBlockQ : kBlockQ * kBlockK;
};

template <int HEAD_DIM>
FelixStatus
launch_sm80_flash_attn_f16_mma16816_v2(half *Q, half *K, half *V, half *O,
                                       uint32_t heads, uint32_t batch_size,
                                       uint32_t seq_len, cudaStream_t stream) {
  using Traits = Sm80FlashAttnF16V2Traits<HEAD_DIM>;

  if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr ||
      heads == 0 || batch_size == 0 || seq_len == 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "SM80 FlashAttention v2 requires non-null Q/K/V/O and non-zero "
        "heads/batch/seq_len");
  }

  if ((seq_len % Traits::kSeqLenMultiple) != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "SM80 FlashAttention v2 requires seq_len aligned to the configured "
        "Q/K tile sizes; tail handling is not implemented yet");
  }

  const int heads_i = static_cast<int>(heads);
  const int batch_size_i = static_cast<int>(batch_size);
  const int seq_len_i = static_cast<int>(seq_len);
  constexpr int head_dim_i = HEAD_DIM;

  sm80_flash_attn_v2::FlashFwdParams<HEAD_DIM> params{
      .q = Q,
      .k = K,
      .v = V,
      .o = O,
      .batch_size = batch_size_i,
      .seqlen_q = seq_len_i,
      .seqlen_k = seq_len_i,
      .heads_q = heads_i,
      .heads_k = heads_i,
      .q_heads_per_kv_head = 1,
      .q_batch_stride = heads_i * seq_len_i * head_dim_i,
      .q_row_stride = head_dim_i,
      .q_head_stride = seq_len_i * head_dim_i,
      .k_batch_stride = heads_i * seq_len_i * head_dim_i,
      .k_row_stride = head_dim_i,
      .k_head_stride = seq_len_i * head_dim_i,
      .v_batch_stride = heads_i * seq_len_i * head_dim_i,
      .v_row_stride = head_dim_i,
      .v_head_stride = seq_len_i * head_dim_i,
      .o_batch_stride = heads_i * seq_len_i * head_dim_i,
      .o_row_stride = head_dim_i,
      .o_head_stride = seq_len_i * head_dim_i,
      .softmax_scale_log2 =
          1.4426950408889634074f / std::sqrt(static_cast<float>(HEAD_DIM)),
  };

  cudaError_t err =
      sm80_flash_attn_v2::fav2_sm80::launch_flash_attn_v2_sm80<HEAD_DIM>(
          params, stream);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }

  return {};
}

} // namespace felix::detail
