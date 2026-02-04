#include "ampere_flash_attn_mma16168_64.cuh"
#include <concepts>
#include <cstdint>
#include <torch/torch.h>
#include "../common.hpp"

namespace sion {
// template<int HEAD_DIM, int STAGE>
struct FlashAttnMMA2DTiling {
};
struct FlashAttnMMA1DTiling {
};
template <typename Traits>
void launch_flash_attn_mma_stages(const torch::Tensor &Q,const torch::Tensor &K,
                                  const torch::Tensor &V, torch::Tensor &O) {

    constexpr auto HEAD_DIM = Traits::Args::HEAD_DIM;
    constexpr auto STAGE = Traits::Args::STAGE;
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kHalf, "Q must be half");
    TORCH_CHECK(K.dtype() == torch::kHalf, "K must be half");
    TORCH_CHECK(V.dtype() == torch::kHalf, "V must be half");
    TORCH_CHECK(O.dtype() == torch::kHalf, "O must be half");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4 && O.dim() == 4,"All tensors must be 4D");

    // B H N D
    auto B = (uint32_t)Q.size(0);
    auto H = (uint32_t)Q.size(1);
    auto N = (uint32_t)Q.size(2);
    auto D = (uint32_t)Q.size(3);

    if constexpr (std::same_as<typename Traits::strategy, FlashAttnMMA2DTiling>) {
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t MMA_M = 16;
    constexpr uint32_t MMA_K = 16;
    constexpr uint32_t MMA_N = 8;
    constexpr uint32_t WARP_ITER_SEQLEN_QS = 2;
    constexpr uint32_t WARP_ITER_SEQLEN_K = 2;
    constexpr uint32_t WARP_NUM_SEQLEN_K = 4;
    constexpr uint32_t WARP_NUM_SEQLEN_QS = 2;
    constexpr uint32_t Br = 64;
    constexpr uint32_t Bc = 64;
    constexpr uint32_t threads = WARP_NUM_SEQLEN_K * WARP_NUM_SEQLEN_QS * warp_size;

    const uint32_t Q_smem_size = Br * HEAD_DIM;
    const uint32_t K_stages_smem_size = Bc * HEAD_DIM * STAGE;
    const uint32_t V_smem_size = Bc * HEAD_DIM;
    const uint32_t S_smem_size = Bc * Br;

    const uint32_t smem_size = (Q_smem_size + K_stages_smem_size + V_smem_size + S_smem_size) * sizeof(at::Half);

    dim3 block(threads);
    dim3 grid(N / Br, H, B);

    half *dQ = reinterpret_cast<half *>(Q.data_ptr<at::Half>());
    half *dK = reinterpret_cast<half *>(K.data_ptr<at::Half>());
    half *dV = reinterpret_cast<half *>(V.data_ptr<at::Half>());
    half *dO = reinterpret_cast<half *>(O.data_ptr<at::Half>());

    cuda_check(cudaGetLastError(), "CUDA pre-launch error");

    cudaFuncSetAttribute(ampere_flash_attn_mma16168_64_2D_warp_tiling<HEAD_DIM, MMA_M, MMA_N, MMA_K, STAGE>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    ampere_flash_attn_mma16168_64_2D_warp_tiling<HEAD_DIM, MMA_M, MMA_N, MMA_K, STAGE>
            <<<grid, block, smem_size>>>(dQ, dK, dV, dO, H, N);

    cuda_check(cudaGetLastError(), "Kernel launch failed");

    cuda_check(cudaDeviceSynchronize(), "Kernel execution failed (runtime error)");

    cuda_check(cudaGetLastError(), "CUDA post-sync error");
    } else if constexpr (std::same_as<Traits::strategy, FlashAttnMMA1DTiling>) {

    }
}

template void launch_flash_attn_mma_stages<64,2>(const torch::Tensor &Q,const torch::Tensor &K,
                                  const torch::Tensor &V, torch::Tensor &O);

template void launch_flash_attn_mma_stages<128,2>(const torch::Tensor &Q,const torch::Tensor &K,
                                  const torch::Tensor &V, torch::Tensor &O);
}