#include "../kernels/sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d.cuh"
#include <felix/registry.hpp>
#include <felix/status.hpp>

namespace felix {

template <int HEAD_DIM, int STAGE, int Bc>
FelixStatus sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t seq_len, cudaStream_t stream) {
  constexpr uint32_t Br = 64;
  constexpr uint32_t warp_size = 32;
  constexpr uint32_t WARP_NUM_SEQLEN_K = 1;
  constexpr uint32_t WARP_NUM_SEQLEN_QS = 4;

  const uint32_t threads = WARP_NUM_SEQLEN_K * WARP_NUM_SEQLEN_QS * warp_size;

  const uint32_t Q_smem_size = Br * HEAD_DIM;
  const uint32_t K_stages_smem_size = Bc * HEAD_DIM * STAGE;
  const uint32_t V_smem_size = Bc * HEAD_DIM;

  const uint32_t smem_size =
      (Q_smem_size + K_stages_smem_size + V_smem_size) * sizeof(half);

  // TODO: support tail tiles when seq_len is not divisible by Br.
  if ((seq_len % Br) != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "seq_len must be divisible by Br=64; tail handling is not "
        "implemented yet");
  }

  dim3 block(threads);
  dim3 grid(seq_len / Br, heads, batch_size);

  cudaError_t err;

  err = cudaFuncSetAttribute(
      sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_kernel<HEAD_DIM, STAGE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
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
} // namespace felix

template felix::FelixStatus
felix::sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_launch<64, 2, 64>(
    half *, half *, half *, half *, uint32_t, uint32_t, uint32_t, cudaStream_t);
template felix::FelixStatus
felix::sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_launch<128, 2, 64>(
    half *, half *, half *, half *, uint32_t, uint32_t, uint32_t, cudaStream_t);

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d,
    felix::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d",
        felix::sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_launch<64, 2, 64>,
        true, {.min_cc = 80, .max_cc = 89, .priority = 50},
        {.head_dim = 64,
         .block_q = 64,
         .block_k = 64,
         .seq_len_multiple = 64}));

REGISTER_KERNEL(
    sm80_flash_attn_f16_hd128_bq64_bk64_mma16168_s2_1d,
    felix::make_flash_attn_kernel(
        "sm80_flash_attn_f16_hd128_bq64_bk64_mma16168_s2_1d",
        felix::sm80_flash_attn_f16_bq64_bk64_mma16168_s2_1d_launch<128, 2, 64>,
        true, {.min_cc = 80, .max_cc = 89, .priority = 50},
        {.head_dim = 128,
         .block_q = 64,
         .block_k = 64,
         .seq_len_multiple = 64}));
