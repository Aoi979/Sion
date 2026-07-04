#pragma once

#include "../kernels/sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d.cuh"
#include <felix/status.hpp>

namespace felix::detail {

constexpr uint32_t kSm80FlashAttnF16Bq64Bk64BlockQ = 64;
constexpr uint32_t kSm80FlashAttnF16Bq64Bk64BlockK = 64;
constexpr uint32_t kSm80FlashAttnF16Bq64Bk64Threads = 128;
constexpr uint32_t kSm80FlashAttnF16Bq64Bk64OptInSmemBytes = 98304;

template <int HEAD_DIM, int STAGE, int Bc>
FelixStatus launch_sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  constexpr uint32_t Br = kSm80FlashAttnF16Bq64Bk64BlockQ;

  constexpr uint32_t Q_smem_size = Br * HEAD_DIM;
  constexpr uint32_t K_stages_smem_size = Bc * HEAD_DIM * STAGE;
  constexpr uint32_t V_smem_size = Bc * HEAD_DIM;
  constexpr uint32_t smem_size =
      (Q_smem_size + K_stages_smem_size + V_smem_size) * sizeof(half);

  if ((seq_len % Br) != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "seq_len must be divisible by Br=64; tail handling is not "
        "implemented yet");
  }

  dim3 block(kSm80FlashAttnF16Bq64Bk64Threads);
  dim3 grid(seq_len / Br, heads, batch_size);

  cudaError_t err = cudaFuncSetAttribute(
      sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_kernel<HEAD_DIM, STAGE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSm80FlashAttnF16Bq64Bk64OptInSmemBytes);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, err);
  }

  sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_kernel<HEAD_DIM, STAGE>
      <<<grid, block, smem_size, stream>>>(Q, K, V, O, heads, seq_len);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }

  return {};
}

} // namespace felix::detail
