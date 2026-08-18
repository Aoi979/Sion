#include "../kernels/sm90_hgemm_f16_nn_m128n256k64_cooperative.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>

namespace cuda_ops_core {
namespace {

Status validate_sm90_hgemm_scalars(float alpha, float beta) {
  if (alpha != 1.0f || beta != 0.0f) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm90 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return {};
}

Status convert_sm90_hgemm_status(cudaError_t err) {
  if (err == cudaSuccess) {
    return {};
  }
  return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, err);
}

} // namespace

Status sm90_hgemm_f16_nn_m128n256k64_cooperative_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_sm90_hgemm_scalars(alpha, beta);
  if (!status.ok()) {
    return status;
  }
  return convert_sm90_hgemm_status(
      sm90_hgemm_cooperative::launch_hgemm_128x256x64_cooperative(
          A, B, C, static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(K), stream));
}

} // namespace cuda_ops_core

REGISTER_KERNEL(sm90_hgemm_f16_nn_m128n256k64_cooperative,
                cuda_ops_core::make_hgemm_kernel(
                    "sm90_hgemm_f16_nn_m128n256k64_cooperative",
                    cuda_ops_core::sm90_hgemm_f16_nn_m128n256k64_cooperative_launch,
                    false, {.min_cc = 90, .priority = 180},
                    {.layout = cuda_ops_core::KernelLayout::NN,
                     .align_m = 128,
                     .align_n = 256,
                     .align_k = 64,
                     .requires_alpha_one_beta_zero = true}));
