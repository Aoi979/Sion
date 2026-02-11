#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <felix/status.hpp>
namespace felix {
FelixStatus ampere_sgemm_64x64_nn_kernel_launch(uint32_t M, uint32_t N, uint32_t K,
                                         float alpha,
                                         float const *A,
                                         float const *B,
                                         float beta, float *C,
                                         cudaStream_t stream);

template<int const HEAD_DIM, int const STAGE, int const Bc = 64>
FelixStatus ampere_flash_attn_mma16168_64_1D_warp_tiling_kernel_launch(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t QKV_seqlen,
    cudaStream_t stream);


} // namespace felix