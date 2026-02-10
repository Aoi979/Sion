#include "ampere_flash_attn_mma16168_64_1D_warp_tiling.cuh"
#include <felix/felix.hpp>

namespace felix {

template <int HEAD_DIM, int STAGE, int Bc>
FelixStatus ampere_flash_attn_mma16168_64_1D_warp_tiling_kernel_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t seq_len,
    cudaStream_t stream) {
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

  dim3 block(threads);
  dim3 grid(seq_len / Br, heads);

  cudaError_t err;

  err = cudaFuncSetAttribute(
      ampere_flash_attn_mma16168_64_1D_warp_tiling<HEAD_DIM, STAGE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, err);
  }

  ampere_flash_attn_mma16168_64_1D_warp_tiling<HEAD_DIM, STAGE>
      <<<grid, block, smem_size, stream>>>(Q, K, V, O, heads, seq_len);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }

  return {};
}
} // namespace felix
template felix::FelixStatus
felix::ampere_flash_attn_mma16168_64_1D_warp_tiling_kernel_launch<64, 2>(
    half *, half *, half *, half *, uint32_t, uint32_t, cudaStream_t);
template felix::FelixStatus
felix::ampere_flash_attn_mma16168_64_1D_warp_tiling_kernel_launch<128, 2>(
    half *, half *, half *, half *, uint32_t, uint32_t, cudaStream_t);
