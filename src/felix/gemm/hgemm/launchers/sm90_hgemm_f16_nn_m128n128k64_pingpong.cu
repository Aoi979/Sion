#include "../kernels/sm90_hgemm_f16_nn_m128n128k64_pingpong.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {
namespace {

FelixStatus validate_sm90_hgemm_scalars(float alpha, float beta) {
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm90 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return {};
}

FelixStatus convert_sm90_hgemm_status(cudaError_t err) {
  if (err == cudaSuccess) {
    return {};
  }
  return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
}

} // namespace

FelixStatus sm90_hgemm_f16_nn_m128n128k64_pingpong_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_sm90_hgemm_scalars(alpha, beta);
  if (!status.ok()) {
    return status;
  }
  return convert_sm90_hgemm_status(
      sm90_hgemm_pingpong::launch_hgemm_128x128x64_pingpong(
          A, B, C, static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(K), stream));
}

} // namespace felix

REGISTER_KERNEL(sm90_hgemm_f16_nn_m128n128k64_pingpong,
                felix::make_hgemm_kernel(
                    "sm90_hgemm_f16_nn_m128n128k64_pingpong",
                    felix::sm90_hgemm_f16_nn_m128n128k64_pingpong_launch, true,
                    {.min_cc = 90, .priority = 200},
                    {.layout = felix::KernelLayout::NN,
                     .align_m = 128,
                     .align_n = 128,
                     .align_k = 64,
                     .requires_alpha_one_beta_zero = true}));
