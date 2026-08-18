#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_ops_core/status.hpp>
#include <string>
namespace cuda_ops_core {
Status sgemm_f32_launch(uint32_t M, uint32_t N, uint32_t K,
                                float alpha, float const *A, float const *B,
                                float beta, float *C, cudaStream_t stream);

Status sgemm_f32_launch_by_name(uint32_t M, uint32_t N, uint32_t K,
                                        float alpha, float const *A,
                                        float const *B, float beta, float *C,
                                        cudaStream_t stream,
                                        const std::string &kernel_name);

Status hgemm_f16_launch(uint32_t M, uint32_t N, uint32_t K,
                                float alpha, half const *A, half const *B,
                                float beta, half *C, cudaStream_t stream);

Status hgemm_f16_nt_launch(uint32_t M, uint32_t N, uint32_t K,
                                   float alpha, half const *A, half const *B,
                                   float beta, half *C, cudaStream_t stream);

Status hgemm_f16_launch_by_name(uint32_t M, uint32_t N, uint32_t K,
                                        float alpha, half const *A,
                                        half const *B, float beta, half *C,
                                        cudaStream_t stream,
                                        const std::string &kernel_name);

Status topk_f32_radix_select_launch(
    float const *data, float *out, uint32_t num_slices, uint32_t slice_size,
    uint32_t k, bool largest, cudaStream_t stream,
    const std::string &kernel_name = "cuda_topk_f32_radix_select");

template <int HEAD_DIM, int Bc = 64>
Status
flash_attn_f16_launch(half *Q, half *K, half *V, half *O, uint32_t heads,
                         uint32_t batch_size, uint32_t QKV_seqlen,
                         cudaStream_t stream);

template <int HEAD_DIM, int Bc = 64>
Status flash_attn_f16_launch_by_name(
    half *Q, half *K, half *V, half *O, uint32_t heads, uint32_t batch_size,
    uint32_t QKV_seqlen, cudaStream_t stream, const std::string &kernel_name);

} // namespace cuda_ops_core
