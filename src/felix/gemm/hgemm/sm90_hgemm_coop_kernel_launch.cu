#include "kernels/sm90_hgemm_coop.cuh"
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

FelixStatus sm90_hgemm_128x256x64_cooperative_kernel_launch(
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

} // namespace felix

REGISTER_KERNEL(
    sm90_hgemm_128x256x64_cooperative,
    (felix::KernelEntry{
        felix::KernelType::HGEMM, "sm90_hgemm_128x256x64_cooperative",
        (void *)felix::sm90_hgemm_128x256x64_cooperative_kernel_launch,
        false}));
