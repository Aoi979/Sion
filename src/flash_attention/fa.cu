#include "fa_v2_mma.cuh"
#include <torch/torch.h>
#include "../common.hpp"

namespace sion {
template<int HEAD_DIM, int STAGE>
void launch_flash_attn_mma_stages(torch::Tensor &Q, torch::Tensor &K,
                                  torch::Tensor &V, torch::Tensor &O,
                                  uint32_t seqlen) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be half");
    TORCH_CHECK(K.scalar_type() == torch::kHalf, "K must be half");
    TORCH_CHECK(V.scalar_type() == torch::kHalf, "V must be half");
    TORCH_CHECK(O.scalar_type() == torch::kHalf, "O must be half");

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
    dim3 grid(seqlen / Br, 1, 1);

    half *dQ = reinterpret_cast<half *>(Q.data_ptr<at::Half>());
    half *dK = reinterpret_cast<half *>(K.data_ptr<at::Half>());
    half *dV = reinterpret_cast<half *>(V.data_ptr<at::Half>());
    half *dO = reinterpret_cast<half *>(O.data_ptr<at::Half>());

    cuda_check(cudaGetLastError(), "CUDA pre-launch error");

    cudaFuncSetAttribute(v2_fwd_kernel<1, HEAD_DIM, MMA_M, MMA_N, MMA_K, STAGE>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    v2_fwd_kernel<1, HEAD_DIM, MMA_M, MMA_N, MMA_K, STAGE>
            <<<grid, block, smem_size>>>(dQ, dK, dV, dO, seqlen);

    cuda_check(cudaGetLastError(), "Kernel launch failed");

    cuda_check(cudaDeviceSynchronize(), "Kernel execution failed (runtime error)");

    cuda_check(cudaGetLastError(), "CUDA post-sync error");
}

}